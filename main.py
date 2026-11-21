from __future__ import annotations

import argparse
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("请输入大于 0 的整数")
    return number


def device_count(value: str) -> str | int:
    return "auto" if value == "auto" else positive_int(value)


def local_json(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".json" or not path.is_file():
        raise argparse.ArgumentTypeError(f"请指定已存在的 JSON 文件：{path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练问答模型，并在有测试集时评估。")
    parser.add_argument("--epochs", type=positive_int, default=50, help="训练轮数，默认 50")
    parser.add_argument("--batch-size", type=positive_int, help="每批样本数，默认读取 YAML")
    parser.add_argument("--data-path", type=local_json, metavar="FILE", help="本次使用的本地 JSON 数据集")
    parser.add_argument("--num-workers", type=int, help="每个数据加载器的工作进程数，默认读取 YAML")
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu", "cuda", "mps"),
                        default="auto", help="训练设备类型，默认自动选择")
    parser.add_argument("--devices", type=device_count, default="auto", metavar="auto|N",
                        help="设备数量，传入正整数或 auto，默认 auto")
    parser.add_argument("--seed", type=int, help="训练随机种子，默认沿用当前随机状态")
    parser.add_argument("--fast-dev-run", action="store_true", help="各阶段最多运行一个批次，不保存检查点")
    parser.add_argument("--skip-test", action="store_true", help="训练完成后跳过测试集评估")
    args = parser.parse_args(argv)
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers 必须大于或等于 0")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed 必须在 0 到 4294967295 之间")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import lightning.pytorch as pl
    from Data.dataload import DataLoad
    from Model.network import NeuralNetwork
    from Tokenizer.tokenizer import Tokenizer

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    tokenizer = Tokenizer()
    dataloader = DataLoad(tokenizer, source_path=args.data_path, batch_size=args.batch_size,
                          num_workers=args.num_workers)
    vocabulary = tokenizer.assign_ids({"train": dataloader.train_dataset.samples})
    network = NeuralNetwork(len(tokenizer), intent_names=list(dataloader.intent_to_id),
                            vocabulary=vocabulary)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(
        network,
        train_dataloaders=dataloader.train_loader,
        val_dataloaders=dataloader.valid_loader if len(dataloader.valid_dataset) else None,
    )
    if not args.skip_test and len(dataloader.test_dataset):
        trainer.test(network, dataloaders=dataloader.test_loader)


if __name__ == "__main__":
    main()
