import asyncio
import os

import torch

from rlinf.scheduler import Cluster, Worker

SEND_GROUP_NAME = "send_worker_group"
RECV_GROUP_NAME = "recv_worker_group"


class SendWorker(Worker):
    def __init__(self):
        super().__init__()
        self.log_info(
            f"SendWorker init rank={self._rank} world_size={self._world_size} "
            f"local_rank={os.environ.get('LOCAL_RANK')} "
            f"visible_devices={os.environ.get('VISIBLE_DEVICES')}"
        )

    def hello_recv(self):
        # 1. Send a message (string or any serializable object) to the RecvWorker group with the same rank as this SendWorker worker.
        msg = f"Hello from SendWorker Rank {self._rank}!"
        self.log_info(f"Rank {self._rank} -> send msg to {self._rank}")
        self.send(msg, dst_group_name=RECV_GROUP_NAME, dst_rank=self._rank)

        # 2. Receive a reply from the RecvWorker group with the same rank.
        self.log_info(f"Rank {self._rank} <- recv reply from {self._rank}")
        reply = self.recv(src_group_name=RECV_GROUP_NAME, src_rank=self._rank)
        self.log_info(f"Rank {self._rank} got reply: {reply}")

        # 3. The send/recv APIs can also handle tensor, list of tensors and dict of tensors.
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dst_rank = (self._rank + 1) % self._world_size  # Send to the next rank
        tensor = torch.ones(
            size=(1, 1), dtype=torch.float32, device=torch.cuda.current_device()
        )
        self.log_info(
            f"Rank {self._rank} -> send tensor to {dst_rank} "
            f"device={tensor.device} shape={tuple(tensor.shape)}"
        )
        self.send(tensor, dst_group_name=RECV_GROUP_NAME, dst_rank=dst_rank)

        tensor_list = [
            torch.tensor(1.0, dtype=torch.float32, device=torch.cuda.current_device())
            for _ in range(4)
        ]
        self.log_info(
            f"Rank {self._rank} -> send tensor_list to {dst_rank} "
            f"len={len(tensor_list)} device={tensor_list[0].device}"
        )
        self.send(tensor_list, dst_group_name=RECV_GROUP_NAME, dst_rank=dst_rank)

        tensor_dict = {
            "tensor1": torch.tensor(
                2.0, dtype=torch.float32, device=torch.cuda.current_device()
            ),
            "tensor2": torch.tensor(
                3.0, dtype=torch.float32, device=torch.cuda.current_device()
            ),
        }
        self.log_info(
            f"Rank {self._rank} -> send tensor_dict to {dst_rank} "
            f"keys={list(tensor_dict.keys())}"
        )
        self.send(tensor_dict, dst_group_name=RECV_GROUP_NAME, dst_rank=dst_rank)

        # 4. Send tensor directly without metadata overhead if you already know the tensor shape and dtype at the recv side.
        tensor = torch.ones(
            size=(2, 1), dtype=torch.float32, device=torch.cuda.current_device()
        )
        self.log_info(
            f"Rank {self._rank} -> send_tensor to {dst_rank} "
            f"device={tensor.device} shape={tuple(tensor.shape)}"
        )
        self.send_tensor(tensor, dst_group_name=RECV_GROUP_NAME, dst_rank=dst_rank)

    def hello_recv_async(self):
        # 1. Send a tensor asynchronously to the RecvWorker group with the next rank.
        dst_rank = (self._rank + 1) % self._world_size
        tensor = torch.ones(
            size=(3, 1), dtype=torch.float32, device=torch.cuda.current_device()
        )
        async_send_work = self.send(
            tensor,
            dst_group_name=RECV_GROUP_NAME,
            dst_rank=dst_rank,
            async_op=True,
        )
        async_send_work.wait()  # Wait for the async send to complete

        # 2. Send a tensor asynchronously and use asyncio to wait for the operation to complete.
        async def send_tensor_async():
            dst_rank = (self._rank + 1) % self._world_size
            tensor = torch.ones(
                size=(4, 1), dtype=torch.float32, device=torch.cuda.current_device()
            )
            async_send_work = self.send(
                tensor,
                dst_group_name=RECV_GROUP_NAME,
                dst_rank=dst_rank,
                async_op=True,
            )
            await async_send_work.async_wait()

        asyncio.run(send_tensor_async())


class RecvWorker(Worker):
    def __init__(self):
        super().__init__()
        self.log_info(
            f"RecvWorker init rank={self._rank} world_size={self._world_size} "
            f"local_rank={os.environ.get('LOCAL_RANK')} "
            f"visible_devices={os.environ.get('VISIBLE_DEVICES')}"
        )

    def hello_recv(self):
        # 1. Receive a message from the SendWorker worker group with the same rank.
        self.log_info(f"Rank {self._rank} <- recv msg from {self._rank}")
        msg = self.recv(src_group_name=SEND_GROUP_NAME, src_rank=self._rank)
        self.log_info(f"Rank {self._rank} got msg: {msg}")

        # 2. Send a reply back to the SendWorker worker group with the same rank.
        reply = f"Hello from RecvWorker Rank {self._rank}!"
        self.log_info(f"Rank {self._rank} -> send reply to {self._rank}")
        self.send(reply, dst_group_name=SEND_GROUP_NAME, dst_rank=self._rank)

        # 3. Receive a tensor, tensor list and tensor dict from the SendWorker worker group with the same rank.
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        src_rank = (self._rank - 1) % self._world_size  # Receive from the previous rank
        self.log_info(f"Rank {self._rank} <- recv tensor from {src_rank}")
        tensor = self.recv(src_group_name=SEND_GROUP_NAME, src_rank=src_rank)
        self.log_info(
            f"Rank {self._rank} got tensor device={tensor.device} shape={tuple(tensor.shape)}"
        )

        self.log_info(f"Rank {self._rank} <- recv tensor_list from {src_rank}")
        tensor_list = self.recv(src_group_name=SEND_GROUP_NAME, src_rank=src_rank)
        self.log_info(
            f"Rank {self._rank} got tensor_list len={len(tensor_list)} device={tensor_list[0].device}"
        )

        self.log_info(f"Rank {self._rank} <- recv tensor_dict from {src_rank}")
        tensor_dict = self.recv(src_group_name=SEND_GROUP_NAME, src_rank=src_rank)
        self.log_info(f"Rank {self._rank} got tensor_dict keys={list(tensor_dict.keys())}")

        # 4. In-place receive tensor directly without metadata overhead.
        tensor = torch.empty(
            size=(2, 1), dtype=torch.float32, device=torch.cuda.current_device()
        )
        self.log_info(
            f"Rank {self._rank} <- recv_tensor from {src_rank} "
            f"device={tensor.device} shape={tuple(tensor.shape)}"
        )
        self.recv_tensor(tensor, src_group_name=SEND_GROUP_NAME, src_rank=src_rank)

    def hello_recv_async(self):
        # 1. Receive a tensor asynchronously from the SendWorker group with the next rank.
        src_rank = (self._rank - 1) % self._world_size
        async_recv_work = self.recv(
            src_group_name=SEND_GROUP_NAME, src_rank=src_rank, async_op=True
        )
        _ = async_recv_work.wait()

        # 2. Receive a tensor asynchronously and use asyncio to wait for the operation to complete.
        async def recv_tensor_async():
            src_rank = (self._rank - 1) % self._world_size
            async_recv_work = self.recv(
                src_group_name=SEND_GROUP_NAME,
                src_rank=src_rank,
                async_op=True,
            )
            _ = await async_recv_work.async_wait()

        asyncio.run(recv_tensor_async())


def main():
    cluster = Cluster(num_nodes=1)
    send_group = SendWorker.create_group().launch(cluster=cluster, name=SEND_GROUP_NAME)
    recv_group = RecvWorker.create_group().launch(cluster=cluster, name=RECV_GROUP_NAME)

    send_group.hello_recv()
    recv_group.hello_recv().wait()

    send_group.hello_recv_async()
    recv_group.hello_recv_async().wait()


if __name__ == "__main__":
    main()
