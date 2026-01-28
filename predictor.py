# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib

# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap

# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt

# 从 LIME 库中导入 LimeTabularExplainer，用于解释表格数据的机器学习模型
from lime.lime_tabular import LimeTabularExplainer

# 加载训练好的随机森林模型（RF.pkl）
model = joblib.load('RF.pkl')

# 从 X_test.csv 文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv('X_test.csv')

# 加载标签编码器
le = joblib.load('label_encoder.pkl')

# 加载训练集中位数用于数据预处理
train_median = joblib.load('train_median.pkl')

# 定义特征名称，对应数据集中的24个列名（根据你的四分类模型）
feature_names = [
    "age",  # 年龄
    "aki",  # 急性肾损伤
    "lc",  # 肝硬化
    "hf",  # 心力衰竭
    "sapsii",  # SAPS II评分
    "hematocrit",  # 血细胞比容
    "hemoglobin",  # 血红蛋白
    "platelet",  # 血小板计数
    "rdw",  # 红细胞分布宽度
    "rbc",  # 红细胞计数
    "wbc",  # 白细胞计数
    "anion_gap",  # 阴离子间隙
    "chloride",  # 氯化物
    "glucose",  # 葡萄糖
    "sodium",  # 钠
    "lac",  # 乳酸
    "creatinine",  # 肌酐
    "bun",  # 血尿素氮
    "hr",  # 心率
    "rr",  # 呼吸频率
    "temperature",  # 体温
    "inr",  # 国际标准化比值
    "pt",  # 凝血酶原时间
    "aptt"  # 活化部分凝血活酶时间
]

# 定义特征描述（用于帮助用户理解）
feature_descriptions = {
    "age": "患者年龄（岁）",
    "aki": "急性肾损伤（0=无，1=有）",
    "lc": "肝硬化（0=无，1=有）",
    "hf": "心力衰竭（0=无，1=有）",
    "sapsii": "简化急性生理学评分II（0-160分）",
    "hematocrit": "血细胞比容（%）",
    "hemoglobin": "血红蛋白（g/dL）",
    "platelet": "血小板计数（×10^9/L）",
    "rdw": "红细胞分布宽度（%）",
    "rbc": "红细胞计数（×10^12/L）",
    "wbc": "白细胞计数（×10^9/L）",
    "anion_gap": "阴离子间隙（mmol/L）",
    "chloride": "氯化物（mmol/L）",
    "glucose": "葡萄糖（mg/dL）",
    "sodium": "钠（mmol/L）",
    "lac": "乳酸（mmol/L）",
    "creatinine": "肌酐（mg/dL）",
    "bun": "血尿素氮（mg/dL）",
    "hr": "心率（次/分）",
    "rr": "呼吸频率（次/分）",
    "temperature": "体温（℃）",
    "inr": "国际标准化比值",
    "pt": "凝血酶原时间（秒）",
    "aptt": "活化部分凝血活酶时间（秒）"
}

# Streamlit 用户界面
st.title("四组疾病分类预测系统")  # 设置网页标题

# 添加侧边栏用于说明
with st.sidebar:
    st.header("关于这个模型")
    st.write("""
    ## 模型信息
    - **算法**: XGBoost + SMOTE
    - **类别数**: 4组
    - **特征数**: 24个
    - **功能**: 根据患者临床指标预测疾病分组
    """)

    st.header("类别说明")
    st.write("""
    - **Group 0**: 低风险组
    - **Group 1**: 中风险组  
    - **Group 2**: 高风险组
    - **Group 3**: 极高风险组
    """)

# 使用两列布局来组织输入
col1, col2 = st.columns(2)

with col1:
    # 第1-12个特征
    age = st.number_input(feature_descriptions["age"], min_value=0, max_value=120, value=60, help="患者年龄")
    aki = st.selectbox(feature_descriptions["aki"], options=[0, 1], format_func=lambda x: "有" if x == 1 else "无")
    lc = st.selectbox(feature_descriptions["lc"], options=[0, 1], format_func=lambda x: "有" if x == 1 else "无")
    hf = st.selectbox(feature_descriptions["hf"], options=[0, 1], format_func=lambda x: "有" if x == 1 else "无")
    sapsii = st.number_input(feature_descriptions["sapsii"], min_value=0, max_value=160, value=40)
    hematocrit = st.number_input(feature_descriptions["hematocrit"], min_value=10.0, max_value=60.0, value=35.0,
                                 step=0.1)
    hemoglobin = st.number_input(feature_descriptions["hemoglobin"], min_value=5.0, max_value=20.0, value=12.0,
                                 step=0.1)
    platelet = st.number_input(feature_descriptions["platelet"], min_value=0, max_value=1000, value=200)
    rdw = st.number_input(feature_descriptions["rdw"], min_value=10.0, max_value=30.0, value=15.0, step=0.1)
    rbc = st.number_input(feature_descriptions["rbc"], min_value=2.0, max_value=8.0, value=4.5, step=0.1)
    wbc = st.number_input(feature_descriptions["wbc"], min_value=0.0, max_value=50.0, value=8.0, step=0.1)
    anion_gap = st.number_input(feature_descriptions["anion_gap"], min_value=0.0, max_value=30.0, value=12.0, step=0.1)

with col2:
    # 第13-24个特征
    chloride = st.number_input(feature_descriptions["chloride"], min_value=80.0, max_value=120.0, value=100.0, step=0.1)
    glucose = st.number_input(feature_descriptions["glucose"], min_value=50.0, max_value=500.0, value=100.0, step=0.1)
    sodium = st.number_input(feature_descriptions["sodium"], min_value=120.0, max_value=160.0, value=140.0, step=0.1)
    lac = st.number_input(feature_descriptions["lac"], min_value=0.0, max_value=20.0, value=1.5, step=0.1)
    creatinine = st.number_input(feature_descriptions["creatinine"], min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    bun = st.number_input(feature_descriptions["bun"], min_value=5.0, max_value=100.0, value=20.0, step=0.1)
    hr = st.number_input(feature_descriptions["hr"], min_value=40, max_value=180, value=80)
    rr = st.number_input(feature_descriptions["rr"], min_value=8, max_value=50, value=18)
    temperature = st.number_input(feature_descriptions["temperature"], min_value=35.0, max_value=42.0, value=37.0,
                                  step=0.1)
    inr = st.number_input(feature_descriptions["inr"], min_value=0.5, max_value=5.0, value=1.2, step=0.1)
    pt = st.number_input(feature_descriptions["pt"], min_value=10.0, max_value=50.0, value=13.0, step=0.1)
    aptt = st.number_input(feature_descriptions["aptt"], min_value=20.0, max_value=100.0, value=35.0, step=0.1)

# 处理输入数据并进行预测
feature_values = [
    age, aki, lc, hf, sapsii,
    hematocrit, hemoglobin, platelet,
    rdw, rbc, wbc,
    anion_gap, chloride, glucose, sodium, lac,
    creatinine, bun,
    hr, rr, temperature,
    inr, pt, aptt
]

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("预测"):
    # 将特征转换为NumPy数组
    features = np.array([feature_values])

    # 转换为DataFrame并进行与训练集相同的预处理
    input_df = pd.DataFrame(features, columns=feature_names)
    input_df = input_df.fillna(train_median).fillna(0).astype(np.float32)

    # 预测类别（0-3）
    predicted_class = model.predict(input_df)[0]

    # 预测类别的概率
    predicted_proba = model.predict_proba(input_df)[0]

    # 将编码后的类别转换为原始标签
    original_label = le.inverse_transform([predicted_class])[0]

    # 显示预测结果
    st.subheader("预测结果")
    st.write(f"**预测分组:** {original_label}")
    st.write(f"**分组编码:** {predicted_class}")

    # 显示所有类别的概率
    st.write("**各分组概率:**")
    for i, prob in enumerate(predicted_proba):
        group_name = le.inverse_transform([i])[0]
        st.write(f"- {group_name}: {prob:.2%}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100

    # 根据分组给出不同的建议
    advice_dict = {
        "Group0": f"根据模型预测，您属于低风险组（Group 0）。模型预测您属于此组的概率为{probability:.1f}%。建议保持健康生活方式，定期体检。",
        "Group1": f"根据模型预测，您属于中风险组（Group 1）。模型预测您属于此组的概率为{probability:.1f}%。建议密切监测相关指标，咨询专科医生。",
        "Group2": f"根据模型预测，您属于高风险组（Group 2）。模型预测您属于此组的概率为{probability:.1f}%。建议立即咨询专科医生，进行进一步检查和干预。",
        "Group3": f"根据模型预测，您属于极高风险组（Group 3）。模型预测您属于此组的概率为{probability:.1f}%。建议立即就医，可能需要住院治疗和密切监护。"
    }

    st.write("**医疗建议:**")
    st.write(advice_dict.get(original_label, "暂无特定建议"))

    # SHAP 解释
    st.subheader("SHAP 解释")

    try:
        # 创建SHAP解释器
        explainer_shap = shap.TreeExplainer(model.named_steps.get('xgb', model))

        # 计算SHAP值
        shap_values = explainer_shap.shap_values(input_df)

        # 显示当前预测类别的force plot
        st.write(f"**{original_label}的SHAP解释:**")
        shap.force_plot(
            explainer_shap.expected_value[predicted_class],
            shap_values[predicted_class][0],
            input_df.iloc[0],
            matplotlib=True
        )

        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png", caption=f'{original_label}的SHAP Force Plot解释')

    except Exception as e:
        st.warning(f"SHAP解释生成失败: {e}")
        st.write("可以尝试使用LIME解释器")

    # LIME Explanation
    st.subheader("LIME 解释")

    try:
        lime_explainer = LimeTabularExplainer(
            training_data=X_test.values,
            feature_names=X_test.columns.tolist(),
            class_names=[str(c) for c in le.classes_],  # 使用实际的类别名称
            mode='classification'
        )

        # 解释当前实例
        lime_exp = lime_explainer.explain_instance(
            data_row=input_df.values.flatten(),
            predict_fn=model.predict_proba
        )

        # 显示LIME解释
        lime_html = lime_exp.as_html(show_table=True)
        st.components.v1.html(lime_html, height=800, scrolling=True)

    except Exception as e:
        st.warning(f"LIME解释生成失败: {e}")

# 在页面底部添加模型信息
st.markdown("---")
with st.expander("查看模型详细信息"):
    st.write("""
    ## 模型训练信息
    - **算法**: XGBoost + SMOTE + BayesSearchCV
    - **类别**: 4分类问题
    - **特征数**: 24个临床指标
    - **调优**: 使用贝叶斯优化进行超参数调优
    - **数据平衡**: 使用SMOTE处理类别不平衡问题

    ## 特征说明
    1. **年龄 (age)**: 患者年龄
    2. **急性肾损伤 (aki)**: 0=无，1=有
    3. **肝硬化 (lc)**: 0=无，1=有
    4. **心力衰竭 (hf)**: 0=无，1=有
    5. **SAPS II评分 (sapsii)**: 简化急性生理学评分II
    6. **血细胞比容 (hematocrit)**: 血液中红细胞的体积百分比
    7. **血红蛋白 (hemoglobin)**: 血液中携带氧气的蛋白质
    8. **血小板计数 (platelet)**: 血液中的血小板数量
    9. **红细胞分布宽度 (rdw)**: 红细胞大小的变异系数
    10. **红细胞计数 (rbc)**: 血液中红细胞的数量
    11. **白细胞计数 (wbc)**: 血液中白细胞的数量
    12. **阴离子间隙 (anion_gap)**: 血液中未测量的阴离子
    13. **氯化物 (chloride)**: 血液中的氯离子浓度
    14. **葡萄糖 (glucose)**: 血液中的糖分浓度
    15. **钠 (sodium)**: 血液中的钠离子浓度
    16. **乳酸 (lac)**: 血液中的乳酸浓度
    17. **肌酐 (creatinine)**: 肾脏功能的指标
    18. **血尿素氮 (bun)**: 肾脏功能的另一个指标
    19. **心率 (hr)**: 每分钟心跳次数
    20. **呼吸频率 (rr)**: 每分钟呼吸次数
    21. **体温 (temperature)**: 体温
    22. **国际标准化比值 (inr)**: 凝血功能的标准化指标
    23. **凝血酶原时间 (pt)**: 凝血功能测试
    24. **活化部分凝血活酶时间 (aptt)**: 凝血功能测试
    """)