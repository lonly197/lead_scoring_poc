"""
JSON 特征提取模块测试

测试从跟进详情_JSON 提取关键业务特征的功能。
"""

import pytest
import pandas as pd

from src.data.json_extractor import extract_followup_features, batch_extract_json_features


class TestExtractFollowupFeatures:
    """测试单条 JSON 特征提取"""

    def test_empty_json(self):
        """测试空 JSON"""
        result = extract_followup_features(None)
        assert result['跟进总次数'] == 0
        assert result['接通次数'] == 0
        assert result['接通率'] == 0.0

    def test_empty_string(self):
        """测试空字符串"""
        result = extract_followup_features('')
        assert result['跟进总次数'] == 0

    def test_single_record(self):
        """测试单条记录"""
        json_str = '{"意向级别": "H", "通话结果": "已接通", "跟进结果": "继续"}'
        result = extract_followup_features(json_str)
        assert result['跟进总次数'] == 1
        assert result['接通次数'] == 1
        assert result['继续跟进次数'] == 1

    def test_multiple_records(self):
        """测试多条记录"""
        json_str = '''
        [
            {"意向级别": "H", "通话结果": "已接通", "跟进结果": "继续"},
            {"意向级别": "A", "通话结果": "未接通", "跟进结果": "继续"},
            {"意向级别": "B", "通话结果": "已接通", "跟进结果": "战败"}
        ]
        '''
        result = extract_followup_features(json_str)
        assert result['跟进总次数'] == 3
        assert result['接通次数'] == 2
        assert result['未接通次数'] == 1
        assert result['最终战败'] == True
        assert result['意向级别下降'] == True

    def test_level_upgrade(self):
        """测试意向级别提升"""
        json_str = '''
        [
            {"意向级别": "B", "通话结果": "已接通"},
            {"意向级别": "A", "通话结果": "已接通"},
            {"意向级别": "H", "通话结果": "已接通"}
        ]
        '''
        result = extract_followup_features(json_str)
        assert result['意向级别下降'] == False

    def test_call_duration(self):
        """测试通话时长解析"""
        json_str = '''
        [
            {"通话时长(分:秒)": "2:30"},
            {"通话时长(分:秒)": "1:45"},
            {"通话时长(分:秒)": "0:30"}
        ]
        '''
        result = extract_followup_features(json_str)
        assert result['最长通话时长_秒'] == 150  # 2*60 + 30
        assert result['平均通话时长_秒'] == pytest.approx(95.0, rel=0.01)  # (150+105+30)/3

    def test_keywords_extraction(self):
        """测试关键词提取"""
        json_str = '''
        [
            {"跟进备忘": "客户询问价格优惠，打算周末来店试驾，对比了比亚迪和特斯拉"}
        ]
        '''
        result = extract_followup_features(json_str)
        assert result['提及价格'] == True
        assert result['提及试驾'] == True
        assert result['提及到店'] == True
        assert result['提及竞品'] == True

    def test_keywords_not_found(self):
        """测试关键词不存在"""
        json_str = '''
        [
            {"跟进备忘": "客户表示暂时不考虑购车"}
        ]
        '''
        result = extract_followup_features(json_str)
        assert result['提及价格'] == False
        assert result['提及试驾'] == False
        assert result['提及到店'] == False
        assert result['提及竞品'] == False

    def test_invalid_json(self):
        """测试无效 JSON"""
        result = extract_followup_features('not a valid json')
        assert result['跟进总次数'] == 0


class TestBatchExtractJsonFeatures:
    """测试批量特征提取"""

    def test_batch_extraction(self):
        """测试批量提取"""
        df = pd.DataFrame({
            '跟进详情_JSON': [
                '[{"意向级别": "H", "通话结果": "已接通"}]',
                None,
                '[{"意向级别": "A", "通话结果": "未接通"}]',
            ]
        })

        result = batch_extract_json_features(df)

        assert '跟进总次数' in result.columns
        assert '接通次数' in result.columns
        assert len(result) == 3

        # 验证提取结果
        assert result.iloc[0]['跟进总次数'] == 1
        assert result.iloc[0]['接通次数'] == 1
        assert result.iloc[1]['跟进总次数'] == 0
        assert result.iloc[2]['接通次数'] == 0

    def test_missing_column(self):
        """测试 JSON 列不存在"""
        df = pd.DataFrame({
            '其他列': [1, 2, 3]
        })

        result = batch_extract_json_features(df)

        # 应该返回原数据框，不做修改
        assert '跟进总次数' not in result.columns

    def test_drop_original(self):
        """测试删除原始列"""
        df = pd.DataFrame({
            '跟进详情_JSON': [
                '[{"意向级别": "H"}]',
            ]
        })

        result = batch_extract_json_features(df, drop_original=True)

        assert '跟进详情_JSON' not in result.columns
        assert '跟进总次数' in result.columns

    def test_custom_column_name(self):
        """测试自定义列名"""
        df = pd.DataFrame({
            'custom_json': [
                '[{"意向级别": "H"}]',
            ]
        })

        result = batch_extract_json_features(df, json_column='custom_json')

        assert '跟进总次数' in result.columns


class TestEdgeCases:
    """测试边界情况"""

    def test_o_level_in_level_order(self):
        """测试 O 级别在级别顺序中"""
        json_str = '''
        [
            {"意向级别": "O"},
            {"意向级别": "H"}
        ]
        '''
        result = extract_followup_features(json_str)
        # O(4) -> H(3) 应该是下降
        assert result['意向级别下降'] == True

    def test_single_level_no_change(self):
        """测试只有一个级别时没有变化"""
        json_str = '''
        [
            {"意向级别": "H"}
        ]
        '''
        result = extract_followup_features(json_str)
        # 只有一个级别，无法判断变化
        assert result['意向级别下降'] == False

    def test_empty_call_duration(self):
        """测试空通话时长"""
        json_str = '''
        [
            {"通话时长(分:秒)": ""},
            {"通话时长(分:秒)": null}
        ]
        '''
        result = extract_followup_features(json_str)
        assert result['平均通话时长_秒'] == 0.0
        assert result['最长通话时长_秒'] == 0