#!/usr/bin/env python3
"""
领域适应性微调训练器
针对金融报告领域进行专门微调
"""

from typing import Dict, Any, List, Tuple
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DomainAdaptationTrainer:
    """
    领域适应性微调训练器
    针对金融报告领域进行专门微调
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化领域适应性微调训练器
        
        Args:
            config: 微调配置
        """
        try:
            # 默认配置
            self.default_config = {
                "domain": "finance",
                "corpus_dir": "./data/finance_corpus",
                "output_dir": "./models/fine_tuned",
                "model_name_or_path": "THUDM/chatglm-6b",
                "max_length": 2048,
                "batch_size": 4,
                "learning_rate": 5e-5,
                "num_train_epochs": 3,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 100
            }
            
            # 合并配置
            self.config = {**self.default_config, **(config or {})}
            
            # 创建输出目录
            os.makedirs(self.config["output_dir"], exist_ok=True)
            
            logger.info("DomainAdaptationTrainer initialized successfully")
            logger.debug(f"Trainer config: {self.config}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DomainAdaptationTrainer: {str(e)}")
            raise
    
    def prepare_finetuning_data(self) -> List[Dict[str, str]]:
        """
        准备金融报告微调数据
        
        Returns:
            List[Dict[str, str]]: 微调样本列表，格式为{"instruction": str, "input": str, "output": str}
        """
        try:
            logger.info("Preparing finetuning data...")
            
            # 加载金融语料
            data = {
                "financial_reports": self._load_financial_corpus("reports"),
                "news_analyses": self._load_financial_corpus("news"),
                "investor_communications": self._load_financial_corpus("earnings_calls")
            }
            
            # 构建指令微调格式
            finetuning_samples = []
            
            # 处理金融报告
            for report in data["financial_reports"][:500]:  # 采样
                sample = self._create_report_sample(report)
                if sample:
                    finetuning_samples.append(sample)
            
            # 处理新闻分析
            for news in data["news_analyses"][:300]:  # 采样
                sample = self._create_news_sample(news)
                if sample:
                    finetuning_samples.append(sample)
            
            # 处理投资者沟通
            for call in data["investor_communications"][:200]:  # 采样
                sample = self._create_earnings_call_sample(call)
                if sample:
                    finetuning_samples.append(sample)
            
            logger.info(f"Prepared {len(finetuning_samples)} finetuning samples")
            
            # 保存微调数据
            data_path = os.path.join(self.config["output_dir"], "finetuning_data.jsonl")
            self._save_finetuning_data(finetuning_samples, data_path)
            
            return finetuning_samples
            
        except Exception as e:
            logger.error(f"Failed to prepare finetuning data: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return []
    
    def _load_financial_corpus(self, corpus_type: str) -> List[str]:
        """
        加载金融语料
        
        Args:
            corpus_type: 语料类型 (reports, news, earnings_calls)
            
        Returns:
            List[str]: 语料列表
        """
        try:
            corpus_path = os.path.join(self.config["corpus_dir"], corpus_type)
            
            if not os.path.exists(corpus_path):
                logger.warning(f"Corpus directory not found: {corpus_path}")
                return []
            
            # 加载所有文本文件
            texts = []
            for file_path in Path(corpus_path).rglob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        texts.append(text)
            
            logger.info(f"Loaded {len(texts)} {corpus_type} texts from {corpus_path}")
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load {corpus_type} corpus: {str(e)}")
            return []
    
    def _create_report_sample(self, report: str) -> Optional[Dict[str, str]]:
        """
        创建金融报告微调样本
        
        Args:
            report: 金融报告文本
            
        Returns:
            Optional[Dict[str, str]]: 微调样本
        """
        if len(report) < 500:
            return None
        
        # 简单的报告摘要任务
        return {
            "instruction": "请简要总结以下金融报告的主要内容和关键发现",
            "input": report,
            "output": self._generate_report_summary(report)  # 这里应该使用教师模型生成，目前使用简单实现
        }
    
    def _create_news_sample(self, news: str) -> Optional[Dict[str, str]]:
        """
        创建新闻分析微调样本
        
        Args:
            news: 新闻文本
            
        Returns:
            Optional[Dict[str, str]]: 微调样本
        """
        if len(news) < 200:
            return None
        
        # 简单的新闻影响分析任务
        return {
            "instruction": "请分析以下新闻对相关金融市场或公司的潜在影响",
            "input": news,
            "output": self._generate_news_analysis(news)  # 这里应该使用教师模型生成，目前使用简单实现
        }
    
    def _create_earnings_call_sample(self, call: str) -> Optional[Dict[str, str]]:
        """
        创建财报电话会议微调样本
        
        Args:
            call: 财报电话会议文本
            
        Returns:
            Optional[Dict[str, str]]: 微调样本
        """
        if len(call) < 300:
            return None
        
        # 简单的关键信息提取任务
        return {
            "instruction": "请从以下财报电话会议中提取公司的关键财务指标和未来展望",
            "input": call,
            "output": self._extract_earnings_info(call)  # 这里应该使用教师模型生成，目前使用简单实现
        }
    
    def _generate_report_summary(self, report: str) -> str:
        """
        生成报告摘要（简单实现，实际应使用教师模型）
        
        Args:
            report: 报告文本
            
        Returns:
            str: 报告摘要
        """
        # 简单的句子提取
        sentences = report.split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 提取前3句和后2句
        if len(sentences) <= 5:
            return " ".join(sentences)
        
        summary = " ".join(sentences[:3] + sentences[-2:])
        return summary[:500] + "..." if len(summary) > 500 else summary
    
    def _generate_news_analysis(self, news: str) -> str:
        """
        生成新闻分析（简单实现，实际应使用教师模型）
        
        Args:
            news: 新闻文本
            
        Returns:
            str: 新闻分析
        """
        # 简单的模板填充
        return f"分析：本文报道了{news[:100]}...该事件可能对相关市场产生以下影响：\n1. 短期市场波动\n2. 投资者情绪变化\n3. 长期政策影响"
    
    def _extract_earnings_info(self, call: str) -> str:
        """
        提取财报信息（简单实现，实际应使用教师模型）
        
        Args:
            call: 财报电话会议文本
            
        Returns:
            str: 提取的财报信息
        """
        # 简单的关键词提取
        keywords = ["营收", "利润", "增长", "下滑", "展望", "预期", "策略"]
        
        extracted_info = []
        for sentence in call.split("\n"):
            if any(keyword in sentence for keyword in keywords):
                extracted_info.append(sentence.strip())
                if len(extracted_info) >= 5:
                    break
        
        return "\n".join(extracted_info[:5])
    
    def _save_finetuning_data(self, samples: List[Dict[str, str]], path: str):
        """
        保存微调数据
        
        Args:
            samples: 微调样本列表
            path: 保存路径
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            logger.info(f"Finetuning data saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save finetuning data: {str(e)}")
    
    def setup_model_and_trainer(self):
        """
        设置模型和训练器
        
        Returns:
            Tuple: (model, tokenizer, trainer)
        """
        try:
            logger.info("Setting up model and trainer...")
            
            # 导入必要的库
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
            from datasets import load_dataset
            
            # 加载模型和分词器
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name_or_path"],
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name_or_path"],
                trust_remote_code=True,
                device_map="auto"
            )
            
            # 加载微调数据
            dataset = load_dataset("json", data_files=os.path.join(self.config["output_dir"], "finetuning_data.jsonl"))
            
            # 数据集预处理
            def preprocess_function(examples):
                inputs = [f"{examples['instruction'][i]}\n{examples['input'][i]}" for i in range(len(examples['instruction']))]
                outputs = examples['output']
                
                model_inputs = tokenizer(inputs, max_length=self.config["max_length"], truncation=True, padding=True)
                labels = tokenizer(outputs, max_length=self.config["max_length"], truncation=True, padding=True)
                
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=self.config["output_dir"],
                per_device_train_batch_size=self.config["batch_size"],
                per_device_eval_batch_size=self.config["batch_size"],
                learning_rate=self.config["learning_rate"],
                num_train_epochs=self.config["num_train_epochs"],
                warmup_ratio=self.config["warmup_ratio"],
                weight_decay=self.config["weight_decay"],
                logging_steps=self.config["logging_steps"],
                save_steps=self.config["save_steps"],
                eval_steps=self.config["eval_steps"],
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False
            )
            
            # 创建训练器
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["train"] if "test" not in tokenized_dataset else tokenized_dataset["test"],
                tokenizer=tokenizer
            )
            
            logger.info("Model and trainer setup completed successfully")
            return model, tokenizer, trainer
            
        except Exception as e:
            logger.error(f"Failed to setup model and trainer: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def train(self):
        """
        执行微调训练
        
        Returns:
            Dict[str, Any]: 训练结果
        """
        try:
            logger.info("Starting finetuning...")
            
            # 准备微调数据
            self.prepare_finetuning_data()
            
            # 设置模型和训练器
            model, tokenizer, trainer = self.setup_model_and_trainer()
            
            # 开始训练
            trainer.train()
            
            # 保存最佳模型
            best_model_path = os.path.join(self.config["output_dir"], "best_model")
            trainer.save_model(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            
            # 生成训练报告
            training_report = self._generate_training_report(trainer)
            
            logger.info("Finetuning completed successfully")
            return {
                "status": "success",
                "best_model_path": best_model_path,
                "training_report": training_report
            }
            
        except Exception as e:
            logger.error(f"Finetuning failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _generate_training_report(self, trainer) -> Dict[str, Any]:
        """
        生成训练报告
        
        Args:
            trainer: HuggingFace Trainer实例
            
        Returns:
            Dict[str, Any]: 训练报告
        """
        try:
            # 收集训练指标
            metrics = trainer.state.log_history
            
            # 提取关键指标
            report = {
                "total_train_epochs": self.config["num_train_epochs"],
                "total_steps": trainer.state.global_step,
                "final_train_loss": None,
                "best_eval_loss": None,
                "learning_rate": self.config["learning_rate"]
            }
            
            # 查找最终训练损失和最佳验证损失
            for metric in metrics:
                if "loss" in metric and "epoch" in metric:
                    report["final_train_loss"] = metric["loss"]
                if "eval_loss" in metric:
                    if report["best_eval_loss"] is None or metric["eval_loss"] < report["best_eval_loss"]:
                        report["best_eval_loss"] = metric["eval_loss"]
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate training report: {str(e)}")
            return {}
    
    def evaluate(self, eval_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        评估微调后的模型
        
        Args:
            eval_data_path: 评估数据路径
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            logger.info("Evaluating fine-tuned model...")
            
            # 导入必要的库
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import load_dataset
            
            # 加载最佳模型
            best_model_path = os.path.join(self.config["output_dir"], "best_model")
            tokenizer = AutoTokenizer.from_pretrained(best_model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(best_model_path, trust_remote_code=True, device_map="auto")
            
            # 加载评估数据
            if eval_data_path:
                dataset = load_dataset("json", data_files=eval_data_path)
            else:
                # 使用部分训练数据作为评估数据
                dataset = load_dataset(
                    "json", 
                    data_files=os.path.join(self.config["output_dir"], "finetuning_data.jsonl"),
                    split="train[:10%]"
                )
            
            # 简单的评估（计算困惑度）
            from transformers import pipeline
            
            # 创建文本生成流水线
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            # 评估指标
            total_loss = 0.0
            total_tokens = 0
            
            # 简单的困惑度计算
            for i, sample in enumerate(dataset["train"]):
                if i >= 100:  # 限制评估样本数量
                    break
                
                try:
                    inputs = tokenizer(sample["input"], return_tensors="pt").to(model.device)
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                    
                    total_loss += loss * inputs["input_ids"].shape[1]
                    total_tokens += inputs["input_ids"].shape[1]
                except Exception as e:
                    logger.error(f"Error evaluating sample {i}: {str(e)}")
                    continue
            
            perplexity = None
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                perplexity = 2 ** avg_loss
            
            logger.info(f"Evaluation completed. Perplexity: {perplexity}")
            
            return {
                "perplexity": perplexity,
                "eval_samples": min(len(dataset["train"]), 100),
                "best_model_path": best_model_path
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return {
                "status": "failed",
                "error": str(e)
            }
