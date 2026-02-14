#!/usr/bin/env python3
"""
参数高效微调（PEFT）训练器
使用LoRA、QLoRA等技术减少微调成本
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import json
import os
import torch

logger = logging.getLogger(__name__)

class PEFTTrainer:
    """
    参数高效微调（PEFT）训练器
    使用LoRA、QLoRA等技术减少微调成本
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化PEFT训练器
        
        Args:
            config: PEFT配置
        """
        try:
            # 默认LoRA配置
            self.default_config = {
                "peft_type": "lora",  # 支持 "lora", "qlora", "prefix_tuning", "p_tuning"
                "lora_config": {
                    "r": 16,  # LoRA秩
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                },
                "model_name_or_path": "THUDM/chatglm-6b",
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16
                },
                "output_dir": "./models/peft_fine_tuned",
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
            
            logger.info("PEFTTrainer initialized successfully")
            logger.debug(f"Trainer config: {self.config}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PEFTTrainer: {str(e)}")
            raise
    
    def prepare_finetuning_data(self, data_path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        准备微调数据
        
        Args:
            data_path: 数据路径
            
        Returns:
            List[Dict[str, str]]: 微调数据
        """
        try:
            logger.info("Preparing finetuning data...")
            
            if data_path and os.path.exists(data_path):
                # 加载指定路径的数据
                with open(data_path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f if line.strip()]
            else:
                # 生成示例数据
                data = self._generate_sample_data()
            
            # 保存数据
            if not data_path:
                data_path = os.path.join(self.config["output_dir"], "finetuning_data.jsonl")
                self._save_data(data, data_path)
            
            logger.info(f"Prepared {len(data)} finetuning samples")
            return data
            
        except Exception as e:
            logger.error(f"Failed to prepare finetuning data: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return []
    
    def _generate_sample_data(self) -> List[Dict[str, str]]:
        """
        生成示例数据
        
        Returns:
            List[Dict[str, str]]: 示例数据
        """
        sample_data = [
            {
                "instruction": "请分析黄金价格的近期走势",
                "input": "黄金价格在过去一周上涨了1.5%，达到1950美元/盎司。同期美元指数下跌了0.8%，地缘政治局势紧张。",
                "output": "黄金价格在过去一周上涨1.5%，主要受到美元指数下跌和地缘政治局势紧张的支撑。美元指数下跌使得黄金对其他货币持有者更具吸引力，而地缘政治风险增加了黄金的避险需求。从技术面来看，黄金价格突破了关键阻力位，短期可能继续上行，但需关注美联储的货币政策信号。"
            },
            {
                "instruction": "请解释什么是GDP",
                "input": "国内生产总值（GDP）是一个国家经济活动的重要指标。",
                "output": "国内生产总值（Gross Domestic Product，GDP）是衡量一个国家或地区在一定时期内（通常为一年）生产的所有最终产品和服务的市场价值总和。它是衡量一个国家经济规模和增长速度的核心指标，反映了经济活动的总体水平。GDP通常按生产法、收入法和支出法三种方法计算，三种方法理论上应该得出一致的结果。"
            },
            {
                "instruction": "请分析利率变化对股票市场的影响",
                "input": "中央银行宣布将基准利率上调25个基点。",
                "output": "利率上调通常会对股票市场产生短期负面影响。当利率上升时，借款成本增加，企业融资成本上升，可能导致利润下降。同时，固定收益产品的收益率提高，使得股票对投资者的吸引力相对下降，资金可能从股票市场流向债券市场。然而，长期影响取决于利率调整的原因和经济基本面：如果是为了控制通货膨胀而进行的适度调整，且经济增长强劲，市场可能会逐渐适应并恢复上行趋势。"
            }
        ]
        
        # 生成更多示例
        for i in range(10):
            sample_data.append({
                "instruction": f"请分析市场数据{i+1}",
                "input": f"市场数据{i+1}显示，相关指标有所变化。",
                "output": f"分析市场数据{i+1}：指标变化反映了当前市场的趋势。建议投资者关注相关因素，做出合理决策。"
            })
        
        return sample_data
    
    def _save_data(self, data: List[Dict[str, str]], path: str):
        """
        保存数据
        
        Args:
            data: 数据
            path: 保存路径
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            logger.info(f"Data saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
    
    def setup_model(self) -> Tuple[Any, Any]:
        """
        设置PEFT模型
        
        Returns:
            Tuple: (model, tokenizer)
        """
        try:
            logger.info("Setting up PEFT model...")
            
            # 导入必要的库
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import (get_peft_model, LoraConfig, 
                             prepare_model_for_kbit_training, 
                             PeftModel, PeftConfig)
            
            # 配置量化（如果使用QLoRA）
            quantization_config = None
            if self.config["peft_type"] == "qlora":
                quantization_config = BitsAndBytesConfig(**self.config["quantization_config"])
                logger.info("Using QLoRA with 4-bit quantization")
            
            # 加载基础模型
            tokenizer = AutoTokenizer.from_pretrained(self.config["model_name_or_path"], trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name_or_path"],
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # 准备模型进行k位训练
            if self.config["peft_type"] in ["lora", "qlora"]:
                model = prepare_model_for_kbit_training(model)
            
            # 配置LoRA
            if self.config["peft_type"] in ["lora", "qlora"]:
                lora_config = LoraConfig(**self.config["lora_config"])
                model = get_peft_model(model, lora_config)
                
                # 打印可训练参数
                model.print_trainable_parameters()
            
            logger.info("PEFT model setup completed successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to setup PEFT model: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def setup_trainer(self, model, tokenizer, train_data):
        """
        设置训练器
        
        Args:
            model: 模型
            tokenizer: 分词器
            train_data: 训练数据
            
        Returns:
            Trainer: 训练器
        """
        try:
            logger.info("Setting up trainer...")
            
            from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
            from datasets import Dataset
            
            # 转换数据格式
            dataset = Dataset.from_list(train_data)
            
            # 数据预处理
            def preprocess_function(examples):
                inputs = [f"{examples['instruction'][i]}\n{examples['input'][i]}" for i in range(len(examples['instruction']))]
                outputs = examples['output']
                
                model_inputs = tokenizer(inputs, max_length=self.config["max_length"], truncation=True, padding="max_length")
                labels = tokenizer(outputs, max_length=self.config["max_length"], truncation=True, padding="max_length")
                
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
                greater_is_better=False,
                fp16=True
            )
            
            # 数据收集器
            data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
            
            # 创建训练器
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            
            logger.info("Trainer setup completed successfully")
            return trainer
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def train(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        执行PEFT微调
        
        Args:
            data_path: 数据路径
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        try:
            logger.info("Starting PEFT finetuning...")
            
            # 准备数据
            train_data = self.prepare_finetuning_data(data_path)
            
            # 设置模型
            model, tokenizer = self.setup_model()
            
            # 设置训练器
            trainer = self.setup_trainer(model, tokenizer, train_data)
            
            # 开始训练
            trainer.train()
            
            # 保存模型
            peft_model_path = os.path.join(self.config["output_dir"], "peft_model")
            trainer.save_model(peft_model_path)
            
            # 生成训练报告
            training_report = self._generate_training_report(trainer)
            
            logger.info("PEFT finetuning completed successfully")
            return {
                "status": "success",
                "peft_model_path": peft_model_path,
                "training_report": training_report
            }
            
        except Exception as e:
            logger.error(f"PEFT finetuning failed: {str(e)}")
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
            trainer: 训练器
            
        Returns:
            Dict[str, Any]: 训练报告
        """
        try:
            # 收集训练指标
            metrics = trainer.state.log_history
            
            # 提取关键指标
            report = {
                "peft_type": self.config["peft_type"],
                "lora_config": self.config["lora_config"],
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
    
    def load_peft_model(self, peft_model_path: str) -> Tuple[Any, Any]:
        """
        加载PEFT模型
        
        Args:
            peft_model_path: PEFT模型路径
            
        Returns:
            Tuple: (model, tokenizer)
        """
        try:
            logger.info(f"Loading PEFT model from {peft_model_path}")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel, PeftConfig
            
            # 加载PEFT配置
            peft_config = PeftConfig.from_pretrained(peft_model_path)
            
            # 加载基础模型
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # 加载PEFT模型
            model = PeftModel.from_pretrained(model, peft_model_path)
            
            # 设置为评估模式
            model.eval()
            
            logger.info("PEFT model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load PEFT model: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def generate_text(self, model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
        """
        使用PEFT模型生成文本
        
        Args:
            model: PEFT模型
            tokenizer: 分词器
            prompt: 提示词
            max_new_tokens: 最大生成的新token数
            
        Returns:
            str: 生成的文本
        """
        try:
            # 使用模型生成文本
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0
            )
            
            # 解码生成的文本
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分（去除提示词）
            if prompt in generated_text:
                generated_text = generated_text.split(prompt)[-1].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate text: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return ""
    
    def evaluate(self, model, tokenizer, eval_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        评估PEFT模型
        
        Args:
            model: PEFT模型
            tokenizer: 分词器
            eval_data: 评估数据
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            logger.info("Evaluating PEFT model...")
            
            # 评估指标
            total_samples = min(len(eval_data), 50)  # 限制评估样本数量
            total_loss = 0.0
            correct = 0
            
            # 设置模型为评估模式
            model.eval()
            
            for i, sample in enumerate(eval_data[:total_samples]):
                try:
                    # 生成结果
                    prompt = f"{sample['instruction']}\n{sample['input']}"
                    generated_text = self.generate_text(model, tokenizer, prompt, max_new_tokens=512)
                    
                    # 简单的正确性检查
                    if any(keyword in generated_text for keyword in sample["output"].split()[:5]):
                        correct += 1
                        
                    # 计算损失
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    labels = tokenizer(sample["output"], return_tensors="pt").input_ids.to(model.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs, labels=labels)
                        loss = outputs.loss.item()
                        total_loss += loss
                        
                except Exception as e:
                    logger.error(f"Error evaluating sample {i}: {str(e)}")
                    continue
            
            # 计算评估指标
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            accuracy = correct / total_samples if total_samples > 0 else 0.0
            perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else 0.0
            
            evaluation_results = {
                "avg_loss": avg_loss,
                "perplexity": perplexity,
                "accuracy": accuracy,
                "eval_samples": total_samples
            }
            
            logger.info(f"Evaluation completed. Results: {evaluation_results}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return {}
    
    def merge_and_save_model(self, peft_model_path: str, save_path: str):
        """
        合并PEFT模型和基础模型并保存
        
        Args:
            peft_model_path: PEFT模型路径
            save_path: 保存路径
        """
        try:
            logger.info(f"Merging and saving model to {save_path}")
            
            # 加载合并模型
            model, tokenizer = self.load_peft_model(peft_model_path)
            
            # 合并模型
            merged_model = model.merge_and_unload()
            
            # 保存合并后的模型
            merged_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            logger.info("Model merged and saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to merge and save model: {str(e)}")
            import traceback
            logger.debug(f"Exception details: {traceback.format_exc()}")
