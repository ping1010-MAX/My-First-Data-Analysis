#导入第三方模块
import pandas as pd
import matplotlib.pyplot as plt


#用pandas打开Titanic.csv
df=pd.read_csv(r'D:\pycharm\python\项目实训\Titanic.csv',encoding='gbk')
print(df.columns.tolist())

#将表里面的2urvived换成survived名字
df=df.rename(columns={'2urvived':'survived'})


#读取表头并打印
print(df.head())


#检查数据是否缺失
print(df.isnull().sum())


#用出现次数最多的港口来填充那两个空值
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])


#看看0和1到底是谁是谁
print(df['Sex'].value_counts())
#男性是比女性多的（真实），测出0表示男，1表示女
df['Sex']=df['Sex'].replace({0:'Male',1:'Female'})

#设置图表的中文字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#创建一个画布，包含左右两张图
fig,axes=plt.subplots(1,2,figsize=(12,5))



#图一
#不同性别的生存能力  0：女  1:男
survived_by_sex=df.groupby('Sex')['survived'].mean()
survived_by_sex.plot(kind='bar',ax=axes[0],color=['lightcoral','skyblue'])
axes[0].set_title('不同性别的生还率')
axes[0].set_ylabel('生还率')

#手动给x轴设置标签，把0和1换成中文
axes[0].set_xticklabels(['女性(Female)','男性(Male)'],rotation=45)


#图二
#不同船舱等级生还率
survived_by_class=df.groupby('Pclass')['survived'].mean()
survived_by_class.plot(kind='bar',ax=axes[1],color=['gold','lightgreen','lightgrey'])
axes[1].set_title('不同船舱等级的生还率')
axes[1].set_ylabel('生还率')
axes[1].set_xticklabels(['一等舱','二等舱','三等舱'],rotation=0)

#调试布局并显示
plt.tight_layout()
plt.show()




