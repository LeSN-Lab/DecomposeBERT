import os
import sys

sys.path.append("../")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

from utils.helper import ModelConfig
from utils.dataset_utils.load_dataset import load_data
from utils.model_utils.load_model import load_model
from utils.model_utils.evaluate import evaluate_model
from utils.prune_utils.prune_head import *

logging.basicConfig(
    filename="head_p.log", level=logging.INFO, format="%(asctime)s %(message)s"
)


# 설정 클래스 정의
class Args:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.compute_entropy = True
        self.compute_importance = True
        self.head_mask = None
        self.num_pruning_steps = 8
        self.name = "YahooAnswersTopics"
        self.batch_size = 16
        self.num_workers = 16


def main():
    # 모델과 토크나이저 로드

    args = Args()

    model_config = ModelConfig(args.name, args.device)
    num_labels = model_config.config["num_labels"]

    model, tokenizer, checkpoint = load_model(model_config)

    # 데이터셋 로드
    train_dataloader, valid_dataloader, test_dataloader = load_data(
        args.name, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # 헤드 중요도 및 엔트로피 계산
    (
        attn_entropy,
        head_importance,
        preds,
        labels,
        per_class_head_importance_list,
    ) = compute_heads_importance(
        model,
        model_config,
        valid_dataloader,
    )

    head_importance_prunning(
        model, model_config, test_dataloader, args.num_pruning_steps, per_class_head_importance_list
    )

    temp_head_importance_score = copy.deepcopy(head_importance).cpu().numpy()
    temp_head_importance_score = total_preprocess_prunehead(temp_head_importance_score)
    total_head_importance_prunning(
        model, model_config, test_dataloader, args.num_pruning_steps, temp_head_importance_score
    )


# 메인 코드 실행 부분
if __name__ == "__main__":
    main()
