import json
import logging
import struct
from socket import AF_INET, SOCK_STREAM, socket
from typing import Dict, Optional

from udao_trace.configuration import SparkConf
from udao_trace.utils.logging import _get_logger

from ..data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_runtime,
    get_non_decision_inputs_for_qs_runtime,
)
from ..utils.params import QType
from .atomic_optimizer import AtomicOptimizer

logger = _get_logger(
    name="server",
    std_level=logging.DEBUG,
    file_level=logging.DEBUG,
    log_file_path="runtime_optimizer.log",
)


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
        self.seed = seed
        self.sc = sc

    def solve_query(self, msg: str) -> str:
        d = parse_msg(msg)
        if d is None:
            return f"parse failed for message\n {msg}"
        request_type = d["RequestType"]
        if request_type == "RuntimeLQP":
            non_decision_input = get_non_decision_inputs_for_q_runtime(d, self.sc)
            po_confs, po_objs = self.ro_q.solve(non_decision_input, seed=self.seed)
        elif request_type == "RuntimeQS":
            non_decision_input = get_non_decision_inputs_for_qs_runtime(
                d, is_lqp=False, sc=self.sc
            )
            po_confs, po_objs = self.ro_qs.solve(non_decision_input, seed=self.seed)
        else:
            raise ValueError(f"Query type {request_type} is not supported")

        return ""

    def setup_server(self, host: str, port: int, debug: bool = False) -> None:
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
                    logger.info(f"Received message: {msg}")
                    if not msg:
                        logger.warning(f"No message received, disconnecting {addr}")
                        break
                    if debug:
                        response = "xxx\n"
                    else:
                        response = self.solve_query(msg)
                    conn.sendall(response.encode("utf-8"))
                    logger.info(f"Sent response: {response}")

                conn.close()
        except Exception as e:
            logger.exception(f"Exception occurred: {e}")
            sock.close()

    def sanity_check(self) -> None:
        for file_name in ["sample_runtime_lqp.txt", "sample_runtime_qs.txt"]:
            with open(f"assets/runtime_samples/{file_name}") as f:
                msg = f.read().strip()
            d = parse_msg(msg)
            if d is None:
                raise ValueError(f"Failed to parse message: {msg}")
            request_type = d["RequestType"]
            if request_type == "RuntimeLQP":
                non_decision_df = self.ro_q.extract_non_decision_df(
                    non_decision_input=get_non_decision_inputs_for_q_runtime(d, self.sc)
                )
                (
                    graph_embeddings,
                    non_decision_tabular_features,
                ) = self.ro_q.extract_non_decision_embeddings_from_df(non_decision_df)
                print(graph_embeddings, non_decision_tabular_features)
                print(graph_embeddings.shape, non_decision_tabular_features.shape)
                # po_confs, po_objs = self.ro_q.solve(
                # non_decision_input, seed=self.seed)
            elif request_type == "RuntimeQS":
                non_decision_df = self.ro_qs.extract_non_decision_df(
                    non_decision_input=get_non_decision_inputs_for_qs_runtime(
                        d, is_lqp=False, sc=self.sc
                    )
                )
                (
                    graph_embeddings,
                    non_decision_tabular_features,
                ) = self.ro_qs.extract_non_decision_embeddings_from_df(non_decision_df)
                print(graph_embeddings, non_decision_tabular_features)
                print(graph_embeddings.shape, non_decision_tabular_features.shape)
                # po_confs, po_objs = self.ro_qs.solve(
                #   non_decision_input, seed=self.seed)
            else:
                raise ValueError(f"Query type {request_type} is not supported")
