# coding=utf-8
import argparse
from model import EDEE
from train import *
from torch.utils.data import DataLoader, RandomSampler
from data import *
logger = logging.getLogger()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_path', type=str, default=r'.\data', help='Dataset path.')
    parser.add_argument('--dataset_name', type=str, default='origin',help='Choose ChFinAnn dataset.')
    parser.add_argument('--output_dir', type=str, default=r'.\output', help='Directory to store output data.')
    parser.add_argument('--cache_dir', type=str, default=r'.\cache', help='Directory to store cache data.')
    parser.add_argument('--exp_name', type=str, default='ab_graph_160', help='Experiment name (e.g., exp1, exp2)')
    parser.add_argument('--exp_root', type=str, default=r'.\exp', help='Root directory for all experiments')
    parser.add_argument('--role_role_num', type=int, default=521, help='Number of classes.')

    parser.add_argument('--seed', type=int, default=2022, help='random seed for initialization')
    parser.add_argument('--cuda_id', type=str, default='0', help='Choose which GPUs to run')

    # Model parameters
    parser.add_argument('--embedding_dir', type=str, default=r'.\cache\embedding', help='Directory storing embeddings')
    parser.add_argument('--word_embedding_dim', type=int, default=768, help='Dimension of embeddings')
    parser.add_argument('--word_type_embedding_dim', type=int, default=82, help='Dimension of word_type embeddings')

    # MLP
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers of bilstm.')
    parser.add_argument('--num_mlps', type=int, default=4, help='Number of mlps in the last of model.')
    parser.add_argument('--final_hidden_size', type=int, default=150, help='Hidden size of mlps.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for embedding.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=160.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=25, help="Log every X updates steps.")

    # Transformer参数
    parser.add_argument('--nhead', type=int, default=5, help='Number of attention heads.')
    parser.add_argument('--transformer_hidden_size', type=int, default=2048, help='Transformer feedforward dimension.')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='Number of transformer layers.')

    # ========== 新增：模块开关参数 ==========
    # 多尺度特征融合
    parser.add_argument('--use_multi_scale_fusion', action='store_true', default=True, help='是否使用轻量级多尺度特征融合')
    parser.add_argument('--num_scales', type=int, default=2, help='多尺度融合的尺度数量（固定为2: kernel 1和3）')
    
    # Transformer编码器
    parser.add_argument('--use_transformer_encoder', action='store_true', default=True, help='是否使用Transformer编码器')
    
    # 图解码器
    parser.add_argument('--use_graph_decoder', action='store_true', default=True, help='是否使用图解码器')
    parser.add_argument('--graph_layers', type=int, default=2, help='图神经网络层数')
    parser.add_argument('--graph_dropout', type=float, default=0.1, help='图神经网络dropout率')

    # 动态特征选择（保留）
    parser.add_argument('--use_dynamic_feature_selection', action='store_false', default=True)

    return parser.parse_args()

def check_args(args):
    '''检查参数配置'''
    logger.info("=== 实验配置参数 ===")
    logger.info(f"实验名称: {args.exp_name}")
    logger.info(f"随机种子: {args.seed}")

    
    # 模块状态检查
    logger.info("=== 模块开关状态 ===")
    logger.info(f"多尺度特征融合: {'✅ 启用' if args.use_multi_scale_fusion else '❌ 禁用'}")
    logger.info(f"Transformer编码器: {'✅ 启用' if args.use_transformer_encoder else '❌ 禁用'}")
    logger.info(f"图解码器: {'✅ 启用' if args.use_graph_decoder else '❌ 禁用'}")
    
    # 参数验证
    if args.use_multi_scale_fusion and args.num_scales != 2:
        logger.warning("轻量级多尺度模块固定使用2个尺度(kernel 1和3)，num_scales参数将被忽略")
    
    if not args.use_transformer_encoder and not args.use_multi_scale_fusion:
        logger.warning("同时禁用Transformer和多尺度模块，模型将仅使用基础特征")
    
    logger.info("=" * 50)

def get_collate_fn():
    return my_collate

def main():
    # Setup logging
    for h in logger.handlers:
        logger.removeHandler(h)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                       datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # Parse args
    args = parse_args()
    
    # 构建实验目录结构
    args.exp_dir = os.path.join(args.exp_root, args.exp_name)
    args.output_dir = os.path.join(args.exp_dir, 'output')
    args.tensorboard_dir = os.path.join(args.exp_dir, 'tensorboard')
    
    # 创建所有必要的目录
    os.makedirs(args.exp_dir, exist_ok=True)
    # os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    # 保存实验配置到文件
    config_file = os.path.join(args.exp_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    check_args(args)

    # Setup CUDA, GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('设备: %s', args.device)

    # Set seed
    set_seed(args)

    # Load datasets and vocabs
    train_dataset, train_labels_weight, dev_dataset, dev_labels_weight, word_vocab, wType_tag_vocab = load_datasets_and_vocabs(args)

    # Build Model
    model = EDEE(args, wType_tag_vocab['len'])
    model.to(args.device)

    # 新增：加载已有模型参数（继续训练）
    # args.best_model_dir = os.path.join(args.exp_dir, 'best_model')  # 最佳模型保存目录
    # resume_path = r"E:\eventexarctmodel\a模型主代码_原始数据集\exp\full_更正\best_model\best_model_epoch129_f10.8901.pth"
    # if best_model_files:
    #     resume_path = max(best_model_files, key=os.path.getctime)  # 取最新保存的最佳模型
    # else:
    #     resume_path = None  # 若不存在则从头训练

    #此处为加载断点继续训练功能，暂时不需要，注释掉后会有变量名未定义，所以保留，运行过程会自行跳过
    resume_path = r"E:\eventexarctmodel\a模型主代码_原始数据集\exp\full_更正\best_model\best_model_epoch130_f10.8901.pth"
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 注意：需要先定义optimizer和scheduler再加载它们的状态
        train_sampler = RandomSampler(train_dataset)
        collate_fn = get_collate_fn()
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

        # （将原来定义optimizer和scheduler的代码移到这里之前）
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=int(args.num_train_epochs),
            steps_per_epoch=len(train_dataloader)  # 这里需要先定义train_dataloader
        )

        # 加载优化器和调度器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']  # 获取上次结束的轮次
        logger.info(f"✅ 已加载检查点，从第{start_epoch+1}轮继续训练")
    else:
        logger.info("❌ 未找到检查点，从头开始训练")
        start_epoch=0

    # 记录模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")
    
    # 记录模块详细状态
    module_status = model.get_module_status()
    logger.info("=== 模块详细状态 ===")
    for module_name, status in module_status.items():
        if module_name == 'multi_scale_weights':
            logger.info("多尺度注意力权重:")
            for scale, weight in status.items():
                logger.info(f"  {scale}: {weight:.4f}")
        else:
            logger.info(f"{module_name}: {'✅ 启用' if status else '❌ 禁用'}")

    # Train
    train(args, model, train_dataset, dev_dataset, train_labels_weight, dev_labels_weight,start_epoch=start_epoch)

    # 保存模型
    model_save_dir = os.path.join(args.exp_dir, 'checkpoints')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'model_parameters.pth')
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"模型参数已保存到: {model_save_path}")

if __name__ == "__main__":
    main()