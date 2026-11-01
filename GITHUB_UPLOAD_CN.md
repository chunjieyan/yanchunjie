# GitHub 上传与运行说明

本目录是论文配套代码仓库的上传内容。请将此目录作为一个独立仓库；不要在原始工作区根目录执行 `git add .`。原始数据、逐用户预测面板、模型权重、服务器配置、依赖运行时和审稿草稿均不在本包内。

## 上传前检查

```sh
python tools/validate_release.py
python tools/reproduce.py source-integrity
python tools/reproduce.py preflight
```

这些命令只检查文件、Python 语法、入口帮助和实验矩阵，不会下载数据或运行训练。`preflight` 会如实报告本机缺少的可选依赖。GPU 全流程需要原服务器路径、阶段配置、数据和权重；离线检查通过不代表 42 个模型重新训练通过。

## 创建并上传仓库

在 GitHub 创建一个空仓库，不要自动生成 README 或许可证；本目录已包含 README，原项目许可证尚待作者选择，不能把第三方 MIT 声明直接当作本项目许可证。将终端切换到本目录后执行：

```sh
git init -b main
git add .
git diff --cached --stat
git commit -m "Add KAIS code and reproducibility materials"
```

随后使用 GitHub 页面提供的真实仓库地址执行 `git remote add origin` 和 `git push -u origin main`。本包未创建远程仓库、推送代码或注册 DOI。用户未提供的仓库地址、作者 ORCID、论文 DOI 和开源许可证均未虚构。

`.gitattributes` 保留源文件原始换行，避免 Windows Git 换行转换使 SHA256 校验失效。检查 GitHub Actions 的离线检查结果后，再将真实仓库链接写入论文的 Code availability。若有意增加/修改仓库文件，可先用 `python tools/update_manifest.py` 查看差异，确认后执行 `python tools/update_manifest.py --write` 并重新检查。此命令不改写科学源文件在 provenance 中的原始指纹；不要用重算清单掩盖非预期修改。

## 最先可以运行的任务

```sh
python -m pip install -r requirements.txt
python tools/reproduce.py redraw --output-dir outputs/figures
```

图 1 至图 7 从已保存的最小汇总输入生成，无需数据或 GPU。需本机安装 Arial；可通过 `--font-dir` 指定合法本机字体目录，字体文件不随包分发。`requirements.txt` 是 CPU 绘图与统计依赖范围，不是原 GPU 训练的锁定环境。

查看完整实验来源与42训练/54评估矩阵：

```sh
python tools/replay_training.py --plan --output outputs/training_plan.json
python tools/replay_evaluation.py --plan --output outputs/evaluation_plan.json
python tools/replay_pipeline.py --write-registry outputs/run_registry_template.json
```

Ensemble 的12组评估复用对应 Control 权重，不应额外计入独立训练数。数据取得、预处理顺序、路径限制、独立输出注册和统计重算详见英文 README 与 `docs/REPRODUCING.md`。

## 验证范围

`VALIDATION.md` 记录本次实际运行的检查以及限制。历史结果汇总和原始源文件指纹随包提供；本次整理没有进行42模型的新GPU训练，也没有下载完整数据集或重新产生逐用户预测。GitHub CI只验证离线发布完整性。
