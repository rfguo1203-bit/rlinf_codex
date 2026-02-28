import asyncio

import torch

from rlinf.scheduler import Channel, Cluster, PackedPlacementStrategy, Worker


class Producer(Worker):
    def __init__(self):
        super().__init__()
        self.log_info(
            f"Producer init rank={self._rank} world_size={self._world_size} "
            f"local_rank={self._local_rank}"
        )

    def produce(self, channel: Channel):
        self.log_info("produce: put string sync")
        channel.put("Hello from Producer")

        tensor = torch.ones(1, device=torch.cuda.current_device())
        self.log_info(f"produce: put tensor sync device={tensor.device} shape={tuple(tensor.shape)}")
        channel.put(tensor)

        self.log_info("produce: put string async and wait()")
        async_work = channel.put("Hello from Producer asynchronously", async_op=True)
        async_work.wait()

        self.log_info("produce: put tensor async and await")
        async_work = channel.put(tensor, async_op=True)

        async def wait_async():
            await async_work.async_wait()

        asyncio.run(wait_async())

        self.log_info("produce: put weighted items")
        channel.put("Hello with weight", weight=1)
        channel.put(tensor, weight=2)


class Consumer(Worker):
    def __init__(self):
        super().__init__()
        self.log_info(
            f"Consumer init rank={self._rank} world_size={self._world_size} "
            f"local_rank={self._local_rank}"
        )

    def consume(self, channel: Channel):
        self.log_info("consume: get sync")
        item = channel.get()
        self.log_info(f"consume: got item type={type(item)}")

        self.log_info("consume: get async and wait()")
        async_work = channel.get(async_op=True)
        async_result = async_work.wait()
        self.log_info(f"consume: got async item type={type(async_result)}")

        self.log_info("consume: get async and await")
        async_work = channel.get(async_op=True)

        async def wait_async():
            result = await async_work.async_wait()
            self.log_info(f"consume: got awaited item type={type(result)}")

        asyncio.run(wait_async())

        self.log_info("consume: get_batch target_weight=3")
        batch = channel.get_batch(target_weight=3)
        self.log_info(f"consume: got batch size={len(batch)}")


def main():
    cluster = Cluster(num_nodes=1)
    channel = Channel.create(name="channel")
    placement = PackedPlacementStrategy(start_hardware_rank=0, end_hardware_rank=0)

    producer = Producer.create_group().launch(
        cluster, name="producer_group", placement_strategy=placement
    )
    consumer = Consumer.create_group().launch(
        cluster, name="consumer_group", placement_strategy=placement
    )

    r1 = producer.produce(channel)
    r2 = consumer.consume(channel)
    r1.wait()
    r2.wait()


if __name__ == "__main__":
    main()
