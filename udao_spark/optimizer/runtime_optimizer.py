import json
import struct
from socket import AF_INET, SOCK_STREAM, socket
from typing import Dict, Optional, Tuple

from udao_trace.configuration import SparkConf
from udao_trace.parser.spark_parser import THETA_P, THETA_S

from ..data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_runtime,
    get_non_decision_inputs_for_qs_runtime,
)
from ..utils.logging import logger
from ..utils.params import QType
from .atomic_optimizer import AtomicOptimizer
from .utils import utopia_nearest


def recv_msg(sock: socket) -> Optional[str]:
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    logger.debug(f"Received message length: {msglen}")

    # Read the message data
    msg_data = recvall(sock, msglen)
    if msg_data is None:
        raise ValueError("Message data is None")
    return msg_data.decode("utf-8")


def recvall(sock: socket, n: int) -> Optional[bytearray]:
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        logger.debug(f"Current data length: {len(data)}, target: {n}")
        packet = sock.recv(n - len(data))
        if not packet:
            logger.warning("Packet is empty, returning None")
            return None
        data.extend(packet)
    logger.debug(f"Received data: {data}, with length: {len(data)}")
    return data


def parse_msg(msg: str) -> Optional[Dict]:
    try:
        d: Dict = json.loads(msg)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse message: {msg}, error: {e}")
        return None
    return d


R_Q: QType = "q_all"
R_QS: QType = "qs_pqp_runtime"


class RuntimeOptimizer:
    @classmethod
    def from_params(
        cls,
        bm: str,
        ag_meta_dict: Dict,
        spark_conf: SparkConf,
        decision_variables_dict: Dict,
        seed: Optional[int] = 42,
    ) -> "RuntimeOptimizer":
        optimizer_dict = {
            q_type: AtomicOptimizer(
                bm=bm,
                model_sign=ag_meta_dict[q_type]["model_sign"],
                graph_model_params_path=ag_meta_dict[q_type]["model_params_path"],
                graph_weights_path=ag_meta_dict[q_type]["graph_weights_path"],
                q_type=q_type,
                data_processor_path=ag_meta_dict[q_type]["data_processor_path"],
                spark_conf=spark_conf,
                decision_variables=decision_variables_dict[q_type],
                ag_path=ag_meta_dict[q_type]["ag_path"],
            )
            for q_type in [R_Q, R_QS]
        }
        return cls(
            ro_q=optimizer_dict[R_Q],
            ro_qs=optimizer_dict[R_QS],
            sc=spark_conf,
            seed=seed,
        )

    def __init__(
        self,
        ro_q: AtomicOptimizer,
        ro_qs: AtomicOptimizer,
        sc: SparkConf,
        seed: Optional[int] = 42,
    ):
        self.ro_q = ro_q
        self.ro_qs = ro_qs
        self.sc = sc
        self.seed = seed

    def get_non_decision_input_and_ro(
        self, msg: str
    ) -> Tuple[Dict, AtomicOptimizer, int]:
        d = parse_msg(msg)
        if d is None:
            raise ValueError(f"Failed to parse message: {msg}")
        request_type = d["RequestType"]
        sc = self.sc
        if request_type == "RuntimeLQP":
            non_decision_input = get_non_decision_inputs_for_q_runtime(d, sc)
            ro = self.ro_q
            n_edges = len(d["LQP"]["links"])
        elif request_type == "RuntimeQS":
            non_decision_input = get_non_decision_inputs_for_qs_runtime(d, False, sc)
            ro = self.ro_qs
            n_edges = len(d["QSPhysical"]["links"])
        else:
            raise ValueError(f"Query type {request_type} is not supported")
        return non_decision_input, ro, n_edges

    def solve_msg(
        self,
        msg: str,
        use_ag: bool,
        ag_model: Optional[str],
        sample_mode: str,
        n_samples: int,
        moo_mode: str,
    ) -> str:
        non_decision_input, ro, n_edges = self.get_non_decision_input_and_ro(msg)
        if n_edges == 0:
            logger.warning("No changes to make when no edges in a graph")
            return "{}"
        po_objs, po_confs = ro.solve(
            non_decision_input,
            seed=self.seed,
            use_ag=use_ag,
            ag_model=ag_model,
            sample_mode=sample_mode,
            n_samples=n_samples,
            moo_mode=moo_mode,
        )
        if po_objs is None or po_confs is None or len(po_objs) == 0:
            logger.warning("NFS for message")
            return "NSF"  # No Solution Found

        if len(po_objs) == 1:
            ret_obj, ret_conf = po_objs[0], po_confs[0]
        else:
            ret_obj, ret_conf = utopia_nearest(po_objs, po_confs)
        if ro.ta.q_type == R_Q:
            ret_dict = {k: v for k, v in zip(THETA_P + THETA_S, ret_conf)}
        elif ro.ta.q_type == R_QS:
            ret_dict = {k: v for k, v in zip(THETA_S, ret_conf)}
        else:
            raise ValueError(f"QType {ro.ta.q_type} is not supported")
        ret_msg = json.dumps(ret_dict)
        logger.debug(f"Return {ret_msg}")
        return ret_msg

    def setup_server(
        self,
        host: str,
        port: int,
        debug: bool,
        use_ag: bool,
        ag_model: Optional[str],
        sample_mode: str,
        n_samples: int,
        moo_mode: str,
    ) -> None:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.bind((host, port))
        sock.listen(1)
        logger.info(f"Server listening on {host}:{port}")

        try:
            while True:
                logger.info(f"Server listening on {host}:{port}")
                conn, addr = sock.accept()
                logger.info(f"Connected by {addr}")

                while True:
                    msg = recv_msg(conn)
                    logger.debug(f"Received message: {msg}")
                    if not msg:
                        logger.warning(f"No message received, disconnecting {addr}")
                        break
                    if debug:
                        response = (
                            json.dumps({THETA_S[0]: "0.34", THETA_S[1]: "512KB"}) + "\n"
                        )
                    else:
                        response = (
                            self.solve_msg(
                                msg,
                                use_ag=use_ag,
                                ag_model=ag_model,
                                sample_mode=sample_mode,
                                n_samples=n_samples,
                                moo_mode=moo_mode,
                            )
                            + "\n"
                        )
                    conn.sendall(response.encode("utf-8"))
                    logger.debug(f"Sent response: {response}")

                conn.close()
        except Exception as e:
            logger.exception(f"Exception occurred: {e}")
            sock.close()

    def sanity_check(
        self,
        file_path: str,
        use_ag: bool,
        ag_model: Optional[str],
        sample_mode: str,
        n_samples: int,
        moo_mode: str,
    ) -> None:
        with open(file_path) as f:
            msg = f.read().strip()
        self.solve_msg(
            msg,
            use_ag=use_ag,
            ag_model=ag_model,
            sample_mode=sample_mode,
            n_samples=n_samples,
            moo_mode=moo_mode,
        )
