import re
import os
import matplotlib.pyplot as plt

def parse_trainlog(trainlog_path):
    steps, avg_losses, ppls, ms_per_batch = [], [], [], []

    with open(trainlog_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(
                r'step\s+(\d+).*?ms/batch\s+([\d.]+).*?avg loss\s+([\d.]+).*?ppl\s+([\d.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                ms_per_batch.append(float(match.group(2)))
                avg_losses.append(float(match.group(3)))
                ppls.append(float(match.group(4)))
    return steps, avg_losses, ppls, ms_per_batch

def parse_evallog(evallog_path):
    eval_steps, eval_losses, eval_ppls = [], [], []

    with open(evallog_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(
                r'\| Eval\s+\d+\s+at step\s+(\d+)\s+\|.*?valid loss\s+([\d.]+)\s+\|\s+valid ppl\s+([\d.]+)', line)
            if match:
                eval_steps.append(int(match.group(1)))
                eval_losses.append(float(match.group(2)))
                eval_ppls.append(float(match.group(3)))
    return eval_steps, eval_losses, eval_ppls

def plot_metrics(train_data, eval_data, output_dir):
    steps, avg_losses, ppls, ms_per_batch = train_data
    eval_steps, eval_losses, eval_ppls = eval_data

    def savefig(name):
        plt.savefig(os.path.join(output_dir, name))
        plt.close()

    # 1. Avg Loss
    plt.figure()
    plt.plot(steps, avg_losses, label='Avg Loss')
    plt.xlabel('Step')
    plt.ylabel('Avg Loss')
    plt.title('Training Avg Loss over Steps')
    plt.grid(True)
    plt.legend()
    savefig('train_avg_loss.png')

    # 2. PPL
    plt.figure()
    plt.plot(steps, ppls, label='Training PPL', color='orange')
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.title('Training PPL over Steps')
    plt.grid(True)
    plt.legend()
    savefig('train_ppl.png')

    # 3. ms/batch
    plt.figure()
    plt.plot(steps, ms_per_batch, label='ms/batch', color='green')
    plt.xlabel('Step')
    plt.ylabel('Milliseconds per Batch')
    plt.title('Batch Time over Steps')
    plt.grid(True)
    plt.legend()
    savefig('train_ms_batch.png')

    # 4. Validation Loss
    if eval_steps:
        plt.figure()
        plt.plot(eval_steps, eval_losses, marker='o', label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss over Eval Steps')
        plt.grid(True)
        plt.legend()
        savefig('eval_loss.png')

        # 5. Validation PPL
        plt.figure()
        plt.plot(eval_steps, eval_ppls, marker='s', color='red', label='Validation PPL')
        plt.xlabel('Step')
        plt.ylabel('Validation PPL')
        plt.title('Validation PPL over Eval Steps')
        plt.grid(True)
        plt.legend()
        savefig('eval_ppl.png')

if __name__ == '__main__':
    # 设置文件路径
    log_dir = 'trained_models/GPT2_M/e2e'  # 你的文件夹路径
    trainlog_path = os.path.join(log_dir, 'training_log.txt')
    evallog_path = os.path.join(log_dir, 'validation_log.txt')

    # 读取数据
    train_data = parse_trainlog(trainlog_path)
    eval_data = parse_evallog(evallog_path)

    # 绘图并保存到该文件夹
    plot_metrics(train_data, eval_data, output_dir=log_dir)
