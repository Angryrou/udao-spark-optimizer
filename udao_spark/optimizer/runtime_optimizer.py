import json
import struct
import time
from socket import AF_INET, SOCK_STREAM, socket
from typing import Dict, Optional, Tuple

from udao_trace.configuration import SparkConf
from udao_trace.parser.spark_parser import THETA_P, THETA_S

from ..data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_runtime,
    get_non_decision_inputs_for_qs_runtime,
)
from ..utils.exceptions import SolutionNotFoundError
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
    logger.debug(f"Received the whole data with length: {len(data)}")
    return data


def parse_msg(msg: str) -> Dict:
    try:
        d: Dict = json.loads(msg)
        return d
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse message: {msg}, error: {e}")
        raise ValueError(f"Failed to parse message: {msg}")


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
        verbose: bool = False,
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
                verbose=verbose,
            )
            for q_type in [R_Q, R_QS]
        }
        return cls(
            ro_q=optimizer_dict[R_Q],
            ro_qs=optimizer_dict[R_QS],
            sc=spark_conf,
            seed=seed,
            verbose=verbose,
        )

    def __init__(
        self,
        ro_q: AtomicOptimizer,
        ro_qs: AtomicOptimizer,
        sc: SparkConf,
        seed: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.ro_q = ro_q
        self.ro_qs = ro_qs
        self.sc = sc
        self.seed = seed
        self.verbose = verbose

    def get_non_decision_input_and_ro(self, d: Dict) -> Tuple[Dict, AtomicOptimizer]:
        request_type = d["RequestType"]
        sc = self.sc
        if request_type == "RuntimeLQP":
            non_decision_input = get_non_decision_inputs_for_q_runtime(d, sc)
            ro = self.ro_q
        elif request_type == "RuntimeQS":
            non_decision_input = get_non_decision_inputs_for_qs_runtime(d, False, sc)
            ro = self.ro_qs
        else:
            raise ValueError(f"Query type {request_type} is not supported")
        return non_decision_input, ro

    def filter_msg(self, d: Dict) -> bool:
        request_type = d["RequestType"]
        if request_type == "RuntimeLQP":
            if len(d["LQP"]["links"]) == 0:
                logger.warning("No changes to make when no edges in a graph")
                return True
            if (
                len(
                    [
                        k
                        for k, v in d["LQP"]["operators"].items()
                        if "Join" in v["className"]
                    ]
                )
                == 0
            ):
                logger.warning("No changes to make when no joins in a LQP")
                return True
        elif request_type == "RuntimeQS":
            if len(d["QSPhysical"]["links"]) == 0:
                logger.warning("No changes to make when no edges in a graph")
                return True
        else:
            raise ValueError(f"Query type {request_type} is not supported")
        return False

    def solve_msg(
        self,
        parsed_dict: Dict,
        use_ag: bool,
        ag_model_dict: Dict[QType, Dict[str, str]],
        sample_mode: str,
        n_samples: int,
        moo_mode: str,
    ) -> str:
        t1 = time.perf_counter_ns()
        if self.filter_msg(parsed_dict):
            return "{}"

        t2 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f"Filtering took {(t2 - t1) // 1e6} ms")

        non_decision_input, ro = self.get_non_decision_input_and_ro(parsed_dict)

        t3 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f"Getting non_decision_input and ro took {(t3 - t2) // 1e6} ms")

        po_objs, po_confs = ro.solve(
            non_decision_input,
            seed=self.seed,
            use_ag=use_ag,
            ag_model=ag_model_dict[ro.ta.q_type],
            sample_mode=sample_mode,
            n_samples=n_samples,
            moo_mode=moo_mode,
        )

        t4 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f"Solving took {(t4 - t3) // 1e6} ms")

        if po_objs is None or po_confs is None or len(po_objs) == 0:
            raise SolutionNotFoundError("No Solution Found")

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

        t5 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f"Preparing return message took {(t5 - t4) // 1e6} ms")

        ret_msg = json.dumps(ret_dict)
        logger.debug(f"Return {ret_msg}")

        t6 = time.perf_counter_ns()
        if self.verbose:
            logger.info(f"Dumping return message took {(t6 - t5) // 1e6} ms")

        return ret_msg

    def setup_server(
        self,
        host: str,
        port: int,
        debug: bool,
        use_ag: bool,
        ag_model_dict: Dict[QType, Dict[str, str]],
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

                dt_dict = {}

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
                        t1 = time.perf_counter_ns()
                        parsed_dict = parse_msg(msg)
                        logger.info(
                            f"Parsed message: {parsed_dict}, "
                            f"took {(time.perf_counter_ns() - t1) // 1e6} ms"
                        )
                        response = (
                            self.solve_msg(
                                parsed_dict,
                                use_ag=use_ag,
                                ag_model_dict=ag_model_dict,
                                sample_mode=sample_mode,
                                n_samples=n_samples,
                                moo_mode=moo_mode,
                            )
                            + "\n"
                        )
                        rt = parsed_dict["RequestType"]
                        request_id = (
                            parsed_dict["QsOptId"]
                            if rt == "RuntimeQS"
                            else parsed_dict["LqpOptId"]
                        )
                        dt_ms = (time.perf_counter_ns() - t1) / 1e6
                        logger.info(f"Request {rt}-{request_id} took {dt_ms:.0f} ms")
                        dt_dict[(rt, request_id)] = dt_ms

                    conn.sendall(response.encode("utf-8"))
                    logger.debug(f"Sent response: {response}")
                conn.close()
                logger.info(f"Finished one session, disconnecting {addr}")
                logger.info(f"Request time: {sorted(dt_dict)}")
        except Exception as e:
            logger.exception(f"Exception occurred: {e}")
            sock.close()

    def sanity_check(
        self,
        file_path: str,
        use_ag: bool,
        ag_model_dict: Dict[QType, Dict[str, str]],
        sample_mode: str,
        n_samples: int,
        moo_mode: str,
    ) -> None:
        with open(file_path) as f:
            msg = f.read().strip()
        parsed_dict = parse_msg(msg)
        self.solve_msg(
            parsed_dict,
            use_ag=use_ag,
            ag_model_dict=ag_model_dict,
            sample_mode=sample_mode,
            n_samples=n_samples,
            moo_mode=moo_mode,
        )
