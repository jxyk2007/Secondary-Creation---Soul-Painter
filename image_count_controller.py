# -*- coding: utf-8 -*-
"""
图片数量智能控制模块
根据视频时长和用户设置智能限制生成的图片数量
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ImageCountController:
    """图片数量控制器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化控制器

        Args:
            config: 配置字典，包含控制参数
        """
        self.config = config or {}

        # 从配置中获取参数，提供默认值
        self.max_images = int(self.config.get('max_images', 6))
        self.min_interval_seconds = float(self.config.get('min_interval_seconds', 5.0))
        self.images_per_minute = float(self.config.get('images_per_minute', 2.0))
        self.auto_limit_mode = self.config.get('auto_limit_mode', 'smart')

        logger.info(f"图片控制器初始化: 最大{self.max_images}张, 最小间隔{self.min_interval_seconds}秒, 每分钟{self.images_per_minute}张")

    def filter_funny_moments(self, funny_moments: List[Dict[str, Any]], video_duration: float) -> List[Dict[str, Any]]:
        """
        根据设置过滤搞笑时刻，控制图片数量

        Args:
            funny_moments: 原始搞笑时刻列表
            video_duration: 视频总时长（秒）

        Returns:
            过滤后的搞笑时刻列表
        """
        if not funny_moments:
            logger.warning("没有找到搞笑时刻")
            return []

        logger.info(f"原始搞笑时刻数量: {len(funny_moments)}, 视频时长: {video_duration:.1f}秒")

        # 按搞笑度排序（降序）
        sorted_moments = sorted(funny_moments, key=lambda x: x.get('funny_score', 0), reverse=True)

        if self.auto_limit_mode == 'smart':
            filtered_moments = self._smart_filter(sorted_moments, video_duration)
        else:
            filtered_moments = self._manual_filter(sorted_moments, video_duration)

        logger.info(f"过滤后搞笑时刻数量: {len(filtered_moments)}")
        return filtered_moments

    def _smart_filter(self, moments: List[Dict[str, Any]], video_duration: float) -> List[Dict[str, Any]]:
        """智能过滤模式"""
        logger.info("使用智能过滤模式")

        # 计算建议的图片数量
        duration_minutes = video_duration / 60.0
        suggested_count_by_time = int(duration_minutes * self.images_per_minute)

        # 综合考虑各种限制
        target_count = min(
            self.max_images,  # 用户设置的最大数量
            suggested_count_by_time,  # 按时长比例计算的数量
            len(moments)  # 实际找到的数量
        )

        logger.info(f"智能建议: 按时长{suggested_count_by_time}张, 最终目标{target_count}张")

        # 如果目标数量为0，至少保留1张最搞笑的
        if target_count == 0 and moments:
            target_count = 1

        # 选择最搞笑的N张
        selected_moments = moments[:target_count]

        # 应用时间间隔过滤
        return self._apply_interval_filter(selected_moments)

    def _manual_filter(self, moments: List[Dict[str, Any]], video_duration: float) -> List[Dict[str, Any]]:
        """手动过滤模式"""
        logger.info("使用手动过滤模式")

        # 严格按照最大数量限制
        selected_moments = moments[:self.max_images]

        # 应用时间间隔过滤
        return self._apply_interval_filter(selected_moments)

    def _apply_interval_filter(self, moments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """应用时间间隔过滤"""
        if not moments:
            return []

        # 按时间顺序排序
        time_sorted = sorted(moments, key=lambda x: x.get('start_time', 0))

        filtered = []
        last_time = -999  # 初始值确保第一个时刻总是被选中

        for moment in time_sorted:
            current_time = moment.get('start_time', 0)

            # 检查与上一个选中时刻的时间间隔
            if current_time - last_time >= self.min_interval_seconds:
                filtered.append(moment)
                last_time = current_time
                logger.debug(f"选择时刻: {current_time:.1f}秒 (间隔: {current_time - last_time:.1f}秒)")
            else:
                logger.debug(f"跳过时刻: {current_time:.1f}秒 (间隔太短: {current_time - last_time:.1f}秒)")

        return filtered

    def get_control_info(self, video_duration: float) -> Dict[str, Any]:
        """获取控制信息，用于UI显示"""
        duration_minutes = video_duration / 60.0
        suggested_count = int(duration_minutes * self.images_per_minute)

        return {
            'video_duration': video_duration,
            'duration_minutes': duration_minutes,
            'max_images': self.max_images,
            'min_interval_seconds': self.min_interval_seconds,
            'images_per_minute': self.images_per_minute,
            'suggested_count_by_time': suggested_count,
            'final_max_count': min(self.max_images, suggested_count),
            'auto_limit_mode': self.auto_limit_mode
        }

    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)

        self.max_images = int(self.config.get('max_images', 6))
        self.min_interval_seconds = float(self.config.get('min_interval_seconds', 5.0))
        self.images_per_minute = float(self.config.get('images_per_minute', 2.0))
        self.auto_limit_mode = self.config.get('auto_limit_mode', 'smart')

        logger.info(f"配置已更新: 最大{self.max_images}张, 最小间隔{self.min_interval_seconds}秒, 每分钟{self.images_per_minute}张")


def test_image_count_controller():
    """测试图片数量控制器"""
    print("=== 图片数量控制器测试 ===\n")

    # 模拟搞笑时刻数据
    funny_moments = [
        {'start_time': 5.0, 'funny_score': 0.9, 'description': '超搞笑表情'},
        {'start_time': 8.0, 'funny_score': 0.85, 'description': '意外摔倒'},
        {'start_time': 12.0, 'funny_score': 0.8, 'description': '搞笑对话'},
        {'start_time': 15.0, 'funny_score': 0.75, 'description': '尴尬瞬间'},
        {'start_time': 20.0, 'funny_score': 0.9, 'description': '爆笑反应'},
        {'start_time': 25.0, 'funny_score': 0.7, 'description': '滑稽动作'},
        {'start_time': 30.0, 'funny_score': 0.85, 'description': '惊喜时刻'},
        {'start_time': 35.0, 'funny_score': 0.8, 'description': '搞笑互动'},
        {'start_time': 40.0, 'funny_score': 0.75, 'description': '幽默时刻'},
        {'start_time': 45.0, 'funny_score': 0.9, 'description': '爆笑结尾'},
    ]

    # 测试不同配置
    test_configs = [
        {
            'name': '默认配置 (57秒视频)',
            'config': {'max_images': 6, 'min_interval_seconds': 5.0, 'images_per_minute': 2.0, 'auto_limit_mode': 'smart'},
            'video_duration': 57.0
        },
        {
            'name': '严格限制 (57秒视频)',
            'config': {'max_images': 4, 'min_interval_seconds': 8.0, 'images_per_minute': 1.5, 'auto_limit_mode': 'smart'},
            'video_duration': 57.0
        },
        {
            'name': '手动模式 (57秒视频)',
            'config': {'max_images': 5, 'min_interval_seconds': 6.0, 'images_per_minute': 3.0, 'auto_limit_mode': 'manual'},
            'video_duration': 57.0
        }
    ]

    for test_config in test_configs:
        print(f"测试: {test_config['name']}")
        print(f"配置: {test_config['config']}")

        controller = ImageCountController(test_config['config'])
        control_info = controller.get_control_info(test_config['video_duration'])

        print(f"控制信息: {control_info}")

        filtered_moments = controller.filter_funny_moments(funny_moments, test_config['video_duration'])

        print(f"过滤结果: {len(filtered_moments)} 个时刻")
        for moment in filtered_moments:
            print(f"  {moment['start_time']}秒: {moment['description']} (评分: {moment['funny_score']})")

        print("-" * 50)

    print("测试完成！")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 运行测试
    test_image_count_controller()