# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import regex as re
import torch
import os
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.du.du_swift_engine import (
    DuSwiftEngine, RemoteAddr)
from vllm.distributed.parallel_state import get_world_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.parallel_state import get_pp_group, get_tp_group, get_dp_group

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    slot_mapping_device: torch.Tensor = None

    @staticmethod
    def make_meta(request_id: str, token_ids: list[int], block_ids: list[int],
                  block_size: int) -> "ReqMeta":
        valid_num_tokens = len(token_ids)
        token_ids_tensor = torch.tensor(token_ids)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids_tensor.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]

        return ReqMeta(
            request_id=request_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
        )

        
@dataclass
class DuSwiftConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, token_ids, block_ids, block_size))


class DuSwiftConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: Optional["KVCacheConfig"] = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self.chunked_prefill: dict[str, Any] = {}

        self._rank = get_world_group().rank \
            if role == KVConnectorRole.WORKER else 0
        self._local_rank = get_world_group().local_rank \
            if role == KVConnectorRole.WORKER else 0
        self._dp_rank = get_dp_group().rank_in_group \
            if role == KVConnectorRole.WORKER else 0
        self._pp_rank = get_pp_group().rank_in_group \
            if role == KVConnectorRole.WORKER else 0
        self._tp_rank = get_tp_group().rank_in_group \
            if role == KVConnectorRole.WORKER else 0
        self._dp_size = get_dp_group().world_size \
            if role == KVConnectorRole.WORKER else 0
        self._pp_size = get_pp_group().world_size \
            if role == KVConnectorRole.WORKER else 0
        self._tp_size = get_tp_group().world_size \
            if role == KVConnectorRole.WORKER else 0

        self.du_swift_engine = DuSwiftEngine(
            local_rank=self._local_rank,
            port_offset=self._rank,
            config=self.config,
            model_config=vllm_config.model_config,
            dp_rank=self._dp_rank,
            pp_rank=self._pp_rank,
            tp_rank=self._tp_rank,
            dp_size=self._dp_size,
            pp_size=self._pp_size,
            tp_size=self._tp_size
        ) if role == KVConnectorRole.WORKER else None
        
        self.parallel_config = vllm_config.parallel_config
        self.model_config = vllm_config.model_config
        self.total_num_hidden_layers = getattr(self.model_config.hf_text_config,
                                              "num_hidden_layers", 0)
        self.pp_size = self.parallel_config.pipeline_parallel_size
        self.tp_size = self.parallel_config.tensor_parallel_size
        self.num_card = self.pp_size * self.tp_size

        self.remote_tp_size = self.config.get_from_extra_config(
            "remote_tp_size", self.tp_size)
        self.remote_pp_size = self.config.get_from_extra_config(
            "remote_pp_size", self.pp_size)
        self.enable_asymmetric_p2p = self.config.get_from_extra_config(
            "enable_asymmetric_p2p", False)
        
        self.remote_num_card = self.remote_tp_size * self.remote_pp_size
        self.multiple_machines_d = 1 if self.remote_num_card > 8 else 0
        self.multiple_machines_p = 1 if self.num_card > 8 else 0

        if self.is_producer and self.multiple_machines_p == 1:
            self.ip_map = {}
            self.duplicate_keys = []
            config_file = os.getenv('IP_CONFIG_FILE')
            if not config_file:
                print("Warning: Please set the IPVNet FILE environment variable for cross machine recognition of the second IP address")
                return
            try:
                with open(config_file, 'r', encoding='utf-8') as file:
                    for line_num, line in enumerate(file, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            ips = line.split()
                            if len(ips) == 2:
                                first_ip, second_ip = ips
                                if first_ip not in self.ip_map:
                                    self.ip_map[first_ip] = second_ip
                            else:
                                print(f"warning: num {line_num} Incorrect format : {line}")
            except Exception as e:
                print(f"Error: Exception occurred while reading configuration file - {e}")


    def get_ip_value(self, key):
        return self.ip_map.get(key)
        

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        # Only consumer/decode loads KV Cache
        if self.is_producer:
            return
        assert self.du_swift_engine is not None

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return
        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            request_id: str,
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [2, num_pages, page_size, xxx] if not
                    using MLA, [num_pages, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache. In shape
                    [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
                    otherwise.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
                request_id (str): request id for log
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            if isinstance(attn_metadata, MLACommonMetadata) or all(isinstance(value, MLACommonMetadata) for value in attn_metadata.values())  or dst_kv_cache_layer.ndim == 3:
                num_pages = dst_kv_cache_layer_shape[0]
                page_size = dst_kv_cache_layer_shape[1]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    num_pages * page_size, -1)
                self.check_tensors_except_dim(dst_kv_cache_layer, src_kv_cache,
                                              0)
                num_token = src_kv_cache.shape[0]
                if len(slot_mapping) == num_token:
                    dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
                else:
                    dst_kv_cache_layer[slot_mapping[:num_token],
                                       ...] = src_kv_cache
                    logger.warning(
                        "🚧src_kv_cache does not match, num_slot:%d, "
                        "num_token:%d, request_id:%s", len(slot_mapping),
                        num_token, request_id)

                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
            else:
                num_pages = dst_kv_cache_layer_shape[1]
                page_size = dst_kv_cache_layer_shape[2]
                dst_kv_cache_layer = dst_kv_cache_layer.reshape(
                    2, num_pages * page_size, -1)
                self.check_tensors_except_dim(dst_kv_cache_layer, src_kv_cache,
                                              1)
                num_token = src_kv_cache.shape[1]
                if len(slot_mapping) == num_token:
                    dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
                else:
                    dst_kv_cache_layer[:, slot_mapping[:num_token],
                                       ...] = src_kv_cache
                    logger.warning(
                        "🚧src_kv_cache does not match, num_slot:%d, "
                        "num_token:%d, request_id:%s", len(slot_mapping),
                        num_token, request_id)

                dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)

        # Get the metadata
        metadata: KVConnectorMetadata = \
            self._get_connector_metadata()
        assert isinstance(metadata, DuSwiftConnectorMetadata)

        if metadata is None:
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoE
                kv_cache = getattr(layer, 'kv_cache', None)
                if kv_cache is None:
                    continue

                kv_cache_layer = kv_cache[ \
                    forward_context.virtual_engine]

                if not envs.VLLM_P2P_ASYNC:
                    kv_cache = self.du_swift_engine.recv_tensor(
                        request.request_id + "#" + layer_name)

                    if kv_cache is None:
                        logger.warning("🚧src_kv_cache is None, %s",
                                    request.request_id)
                        continue
                    inject_kv_into_layer(kv_cache_layer, kv_cache,
                                        request.slot_mapping, request.request_id)
                    tensor_id = request.request_id + "#" + layer_name
                    if tensor_id in self.du_swift_engine.recv_store:
                        tensor = self.du_swift_engine.recv_store.pop(tensor_id, None)
                        self.du_swift_engine.send_request_id_to_tensor_ids.pop(
                                    request.request_id, None)
                        self.du_swift_engine.recv_request_id_to_tensor_ids.pop(
                                    request.request_id, None)
                        addr = 0
                        if isinstance(tensor, tuple):
                            addr, _, _ = tensor
                            self.du_swift_engine.pool.free(addr)
                else:
                    dst_kv_cache_layer_shape = kv_cache_layer.shape
                    if isinstance(attn_metadata, MLACommonMetadata) or all(isinstance(value, MLACommonMetadata) for value in attn_metadata.values()):
                        num_pages = dst_kv_cache_layer_shape[0]
                        page_size = dst_kv_cache_layer_shape[1]
                        assert kv_cache_layer.is_contiguous()
                        dst_kv_cache_layer = kv_cache_layer.reshape(
                            num_pages * page_size, -1)
                    else:
                        num_pages = dst_kv_cache_layer_shape[1]
                        page_size = dst_kv_cache_layer_shape[2]
                        assert kv_cache_layer.is_contiguous()
                        dst_kv_cache_layer = kv_cache_layer.reshape(
                            2, num_pages * page_size, -1)

                    inject_start_index = 0
                    for num in range(self.du_swift_engine.tensor_split_num):
                        kv_cache = self.du_swift_engine.recv_tensor(
                            request.request_id + "#" + layer_name + "#" + str(num))

                        if kv_cache is None:
                            logger.warning("🚧src_kv_cache is None, %s",
                                        request.request_id)
                            continue
                        
                        if isinstance(attn_metadata, MLACommonMetadata) or all(isinstance(value, MLACommonMetadata) for value in attn_metadata.values()):
                            num_token = kv_cache.shape[0]
                            if len(request.slot_mapping) == num_token:
                                dst_kv_cache_layer[request.slot_mapping, ...] = kv_cache
                            else:
                                dst_kv_cache_layer[request.slot_mapping[inject_start_index:inject_start_index + num_token],
                                                ...] = kv_cache
                        else:
                            num_token = kv_cache.shape[1]
                            if len(request.slot_mapping) == num_token:
                                dst_kv_cache_layer[:, request.slot_mapping, ...] = kv_cache
                            else:
                                dst_kv_cache_layer[:, request.slot_mapping[inject_start_index:inject_start_index + num_token],
                                                ...] = kv_cache
                        inject_start_index += num_token
                        
                        # inject_kv_into_layer(kv_cache_layer, kv_cache,
                        #                     request.slot_mapping, request.request_id)
                        tensor_id = request.request_id + "#" + layer_name + "#" + str(num)
                        if tensor_id in self.du_swift_engine.recv_store:
                            tensor = self.du_swift_engine.recv_store.pop(tensor_id, None)
                            self.du_swift_engine.send_request_id_to_tensor_ids.pop(
                                        request.request_id, None)
                            self.du_swift_engine.recv_request_id_to_tensor_ids.pop(
                                        request.request_id, None)
                            addr = 0
                            if isinstance(tensor, tuple):
                                addr, _, _ = tensor
                                self.du_swift_engine.pool.free(addr)
                    dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)


    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return

        assert self.du_swift_engine is not None

        is_mla = isinstance(attn_metadata, MLACommonMetadata) or kv_layer.ndim == 3

        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx)
            if MLA is not used, and (num_pages, page_size, xxx) otherwise.
            """            
            if isinstance(attn_metadata, MLACommonMetadata) or layer.ndim == 3:
                num_pages, page_size = layer.shape[0], layer.shape[1]
                return layer.reshape(num_pages * page_size, -1)[slot_mapping,
                                                                ...]
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping,
                                                               ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, DuSwiftConnectorMetadata)

        if envs.VLLM_ENABLE_TBO or envs.VLLM_P2P_ASYNC:
            for request in connector_metadata.requests:
                request_id = request.request_id
                ip, port = self.parse_request_id(request_id, True)
                remote_address = ip + ":" + str(port + self._rank)
                slot_mapping = request.slot_mapping
                if request.slot_mapping_device is None:
                    request.slot_mapping_device = \
                        request.slot_mapping.pin_memory().to(device=kv_layer.device, non_blocking=True)
                slot_mapping = request.slot_mapping_device
                tbo_evt = torch.cuda.Event(enable_timing=False)
                tbo_evt.record()
                pp_rank = (self.parallel_config.rank //
                        self.parallel_config.tensor_parallel_size) % \
                        self.parallel_config.pipeline_parallel_size
                if (self.pp_size == 1):
                    self.du_swift_engine.p2p_async_send_tensor(request_id + "#" + layer_name,
                                                    (kv_layer, slot_mapping), remote_address, tbo_evt)
                elif (self.pp_size == 2):
                    if (pp_rank == 0):
                        self.du_swift_engine.p2p_async_send_tensor(request_id + "#" + layer_name,
                                                        (kv_layer, slot_mapping), remote_address, tbo_evt)
                        self.du_swift_engine.p2p_async_send_tensor(request_id + "#" + layer_name,
                                                        (kv_layer, slot_mapping), ip + ":" + str(port + self._rank + 4), tbo_evt)
                    else:
                        self.du_swift_engine.p2p_async_send_tensor(request_id + "#" + layer_name,
                                                        (kv_layer, slot_mapping), remote_address, tbo_evt)
                        self.du_swift_engine.p2p_async_send_tensor(request_id + "#" + layer_name,
                                                        (kv_layer, slot_mapping), ip + ":" + str(port + self._rank - 4), tbo_evt)
                elif (self.pp_size == 8):
                    for i in range(8):
                        self.du_swift_engine.p2p_async_send_tensor(request_id + "#" + layer_name,
                                                        (kv_layer, slot_mapping), ip + ":" + str(port + i), tbo_evt)
                else:
                    print("Error: only suppprt pp1 pp2 pp8!!!!!!")
        else:
            for request in connector_metadata.requests:
                request_id = request.request_id
                ip, port = self.parse_request_id(request_id, True)
                p_ip, p_port = self.parse_request_id(request_id, False)
                remote_address = ip + ":" + str(port + self._rank)
                # pd_pair_id = p_ip + ":" + p_port + "_" + ip + ":" + port
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)

                pp_rank = (self.parallel_config.rank // self.parallel_config.tensor_parallel_size
                    ) % self.parallel_config.pipeline_parallel_size
                if (self.multiple_machines_p and self.multiple_machines_d):
                    ip_second = self.get_ip_value(ip)
                    if (self.pp_size == 1):
                        if self._rank < 8:
                            self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                                                        kv_cache, remote_address)
                            self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                                                        kv_cache, str(ip_second) + ":" + str(port + self._rank + 8))
                    elif (self.pp_size == 2):
                        if (pp_rank == 0):
                            self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                                                    kv_cache, remote_address)
                        else:
                            self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                                                        kv_cache, str(ip_second) + ":" + str(port + self._rank))
                    else:
                        logger.error("Error: multiple machines only suppprt pp1tp16 and pp2tp8!!!!!!")
                elif (self.multiple_machines_p and not self.multiple_machines_d):
                    if (self.pp_size == 2):
                        remote_address = ip + ":" + str(port + self._tp_rank)
                        self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                                                        kv_cache, remote_address)
                    else:
                        logger.error("Error: P multiple machines D machine only suppprt P:pp2tp8 D:tp8 !!!!!!")

                elif (not self.multiple_machines_p and not self.multiple_machines_d):
                    # remote_addr = RemoteAddr(pd_pair_id, remote_address, self._rank + self.num_card)
                    self.du_swift_engine.send_tensor_new(request_id, layer_name, kv_cache,
                                                is_mla)
                    # if (self.pp_size == 1):
                    #     self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                    #                                 kv_cache, remote_address)
                    # elif (self.pp_size == 2):
                    #     if (pp_rank == 0):
                    #         self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                    #                                     kv_cache, remote_address)
                    #         self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                    #                                     kv_cache, ip + ":" + str(port + self._rank + 4))
                    #     else:
                    #         self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                    #                                     kv_cache, remote_address)
                    #         self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                    #                                     kv_cache, ip + ":" + str(port + self._rank - 4))
                    # elif (self.pp_size == 8):
                    #     for i in range(8):
                    #         self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                    #                                     kv_cache, ip + ":" + str(port + i))
                    # elif (self.enable_asymmetric_p2p):
                    #     self.du_swift_engine.send_tensor(request_id + "#" + layer_name,
                    #                                 kv_cache, remote_address)
                    # else:
                    #     logger.error("Error: P/D single machine only suppprt multiple tp:: (P: pp2tp4 D:tp8 P:pp8tp1 D:tp8) !!!!!!")
                else:
                        logger.error("Error: not support!!!!!!")
    def wait_for_save(self):
        pass
        # if self.is_producer:
        #     assert self.du_swift_engine is not None
        #     self.du_swift_engine.wait_for_sent()

    def get_finished(
            self, finished_req_ids: set[str],
            **kwargs) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        assert self.du_swift_engine is not None

        forward_context: ForwardContext = get_forward_context()
        return self.du_swift_engine.get_finished(finished_req_ids,
                                                 forward_context)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            return 0, False

        num_external_tokens = (len(request.prompt_token_ids) - 1 -
                               num_computed_tokens)

        if num_external_tokens < 0:
            num_external_tokens = 0

        return num_external_tokens, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request, blocks.get_block_ids()[0])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        meta = DuSwiftConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[new_req.req_id]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                # the request's prompt is chunked prefill
                if num_tokens < len(new_req.prompt_token_ids):
                    # 'CachedRequestData' has no attribute 'prompt_token_ids'
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0], new_req.prompt_token_ids)
                    continue
                # the request's prompt is not chunked prefill
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                continue
            if new_req.req_id in self._requests_need_load:
                meta.add_request(request_id=new_req.req_id,
                                 token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size)
                self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids

            if self.is_producer:
                num_scheduled_tokens = (
                    scheduler_output.num_scheduled_tokens)[req_id]
                num_tokens = (num_scheduled_tokens + num_computed_tokens)
                # assert req_id in self.chunked_prefill
                if  req_id not in self.chunked_prefill:
                    continue
                block_ids = new_block_ids[0]
                if not resumed_from_preemption:
                    block_ids = (self.chunked_prefill[req_id][0] + block_ids)
                prompt_token_ids = self.chunked_prefill[req_id][1]
                # the request's prompt is chunked prefill again
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids,
                                                    prompt_token_ids)
                    continue
                # the request's prompt is all prefilled finally
                meta.add_request(request_id=req_id,
                                 token_ids=prompt_token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)
                self.chunked_prefill.pop(req_id, None)
                continue

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                block_ids = new_block_ids[0]

                meta.add_request(request_id=req_id,
                                 token_ids=token_ids,
                                 block_ids=block_ids,
                                 block_size=self._block_size)

        # Requests loaded asynchronously are not in the scheduler_output.
        # for request_id in self._requests_need_load:
        #     request, block_ids = self._requests_need_load[request_id]
        #     meta.add_request(request_id=request.request_id,
        #                      token_ids=request.prompt_token_ids,
        #                      block_ids=block_ids,
        #                      block_size=self._block_size)

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        self.chunked_prefill.pop(request.request_id, None)

        return False, None

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(
            f"Request id {request_id} does not contain hostname and port")

    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
                s1 == s2
                for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim):
            raise NotImplementedError(
                "Currently, only symmetric TP is supported. Asymmetric TP, PP,"
                "and others will be supported in future PRs.")
