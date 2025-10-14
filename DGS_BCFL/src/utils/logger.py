# -*- coding:utf-8 -*-
# @FileName :logger.py
# @Time :2025/10/10
# @Author :M2883b0
# @Description :统一日志配置模块

import logging
import os
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        name: 日志名称，通常使用模块名
        log_file: 日志文件路径，如果为None则只输出到控制台
        level: 日志级别
    
    Returns:
        logging.Logger: 配置好的logger实例
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 创建formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 如果指定了日志文件，创建文件handler
        if log_file:
            # 确保日志文件目录存在
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

# 默认logger实例
DEFAULT_LOGGER = setup_logger('DGS_BCFL')

# 导出常用的日志级别函数，方便使用
info = DEFAULT_LOGGER.info
debug = DEFAULT_LOGGER.debug
warning = DEFAULT_LOGGER.warning
error = DEFAULT_LOGGER.error
critical = DEFAULT_LOGGER.critical