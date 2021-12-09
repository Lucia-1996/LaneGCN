# LaneGCN 代码学习笔记

## 预处理

**orig**: agent 第 t0(20) 时刻的xy坐标，原始 t0 时刻的位置

**rot**: 一个旋转矩阵，其中的theta旋转角是通过第 19 时刻与第 20 时刻的位置计算得到的，计算某 t-1 时刻转到 t0 时刻的，根据 theta 角计算得到，做旋转变换的时候，`xy*rot+orig`

**feats**: 坐标经旋转变换，转移到以 orig 为原点坐标系下，所有 actor 的前 20 时刻的特征，（actor_num,3,20）3=~~x,y坐标~~ *不是x,y坐标，而是每个x,y坐标与相邻前一个坐标的差值*+一个标志位（是否pad)。

<u>注意：所有actor在最初`def read_argo_data`获取data["trajs"]时有将agent放到最前面，下面同理</u>。

**ctrs**: 坐标经旋转变换，转移到以 orig 为原点坐标系下的，所有actor 第20时刻的位置特征

**gt_preds**: 每个actor（agent+所有其他车辆）未来30时刻的xy坐标，ground-truth

**has_pred**: 每个 actor 未来 30 时刻是否有真实的坐标

trajs: 未做旋转变换的 agent +其他所有车辆的所有时刻的 xy 坐标

steps: 未做旋转变换的 agent +其他所有车辆的所有时刻的时间戳 map

theta：t-1 时刻到t0时刻移动过程的旋转角度

**graph**： **ctrs**: 经过旋转坐标变换处理的 lane 上的 node，lane-orig点周边道路搜索到的lanes,node-在搜索到的lane上每两个相邻的centerlines中间取中心点，（如果centerlines是10个，那这条lane的node为9个），ctr存着所有搜索到的lane的所有node

​			**num_nodes**:当前csv文件中抽象出来的所有lane的所有node的个数

​    		**feats**:经旋转坐标变换处理的所有lanes上相邻centerline计算位移差，也是9个

​			**control**: lane是否有交通管制*9

​			**intersect**： lane是否是路口*9

​			**turn**:lane的转向*9

​			**pre**:前驱，6个矩阵，分别代表经1-6次找前驱可达的node 

​			**suc**:后继，6个矩阵，分别代表经1-6次找后继可达的node

​			**left_pairs**: [i,j]，lane i 和它的左邻居j

​			**right_pairs**: [i,j]，lane i和它的右邻居j

( 有关前驱后继：

遍历所有的lane,每一个lane都有centerline,按照centerline得到node(一般是10个centerline9个node)，给所有node编序号，并在idcs中记录哪些是那条lane_segment的node

对每条lanesegment:

pre:v->u,v是u的后继，对当前lane按照前一个node后一个node排好uv后（一般是8对）找到lane的前驱lane，前驱lane的末尾是当前lane末尾的v，然后找到后继lane，当前lane的末尾node是后继lane开头node的v )

### **graph_gather()**

使用batch时，将一个batch 中所有的graph特征合并

比如batch = 8

合并之前data["graph"]8*10，每一个agent都有一个图

合并之后，10维

![image](https://user-images.githubusercontent.com/95835767/145328812-4f92b5d3-c0a5-474e-983f-5a2bfd355e62.png)

其中，idcs对应每一个batch的node个数的range

ctrs没有区别

feats,turn,control,intersect是8个vector的合并concat

pre，suc，left，right也是合并，但是后边一个所有node的序号都是原来的加上前边所有的node个数，如batch中有900个node，第二个graph的uv从900开始算



## model

![image](https://user-images.githubusercontent.com/95835767/145328821-69b0f7cf-aa5a-4257-a05e-4cf8b88b5059.png)

### self.actor_net

3组1d卷积，每一组都由两个残差块组成；然后使用一个特征金字塔融合多尺度特征，并应用另一个残差块获得输出张量。
![image](https://user-images.githubusercontent.com/95835767/145328834-f5aa31ad-a444-430f-8e23-0bb58c60016b.png)
![image](https://user-images.githubusercontent.com/95835767/145328862-593cbb77-28d5-46d4-8969-7ffdcc760b25.png)


### self.map_net

graph:

idcs list 32，每一个元素的个数是对应scence的所有node个数，整个32list整体计数从0到32个node_num的加和

ctrs list 32,每一个scence中的所有node的坐标

feats 长度：32个node_num的拼接，相邻centerline的位移差

pre list6 32个scence的，某阶pre/suc u 拼接 v也拼接，拼接时第1个scence之后的要加索引，相当于重新分配了32个scence为整体的索引

left

right类似于pre suc只不过没有阶数，只有一组uv
![image](https://user-images.githubusercontent.com/95835767/145328886-244364c4-5b76-4a20-b880-7483eec4be69.png)

![image](https://user-images.githubusercontent.com/95835767/145328898-4b13b1ee-7bc2-4a96-999f-cd96c87fffad.png)

![image](https://user-images.githubusercontent.com/95835767/145328900-9a2d955f-5c73-4a9d-9bf6-695c808dacd0.png)
return：

feat -Y



### self.m2m

基本同 `self.map_net`

### self.a2m self.m2a self.a2a

a2m, m2a, a2a网络结构一致，采用空间注意力机制构成残差块。

![image](https://user-images.githubusercontent.com/95835767/145328922-69894c76-012c-4b75-bb03-30e82ffad82a.png)

![image](https://user-images.githubusercontent.com/95835767/145328912-79f57ad4-6313-4f66-a612-821dfc0ea470.png)

## self.pred_net

actors: 上一层 self.a2a 的输出，128维。

pred:list6,每一个元素都是将actors作为输入经self.pred的输出，60维。

reg = preds结果经shape变换后（num_actor,6,30,2）将60维分为了(30,2)对应每个actor后30时刻的预测位置坐标。

`self.pred`有重复的6个残差块，分别对应最后预测结果的6条预测线。

#### output 

分为两部分

#### reg:

预测了所有actor的后30时刻的位置坐标，并根据置信度为30时刻的位置坐标重新排序。（actor_num，6，30，2）

#### cls

计算每个actor 6种模态的置信度。先将预测轨迹（后30时刻）的起点（第20时刻，真值）和终点（第50时刻）坐标差经`self.dist`网络embedding，然后再和actors(Fusionnet的输出 actor feature)行拼接，经`self.agt`残差块得到128维结果。经`self.cls`得到（num_actor,6）的输出，对应所有actor的6个模态的置信度。

将每个actor的6个预测线的置信度降序排序，并对reg做相应的排序。（actor_num，6）

### loss

根据网络输出output，计算loss



总loss=分类和回归损失总和：loss = cls + reg

cls：

$\hat{k}$是6条预测线中最终位移误差最小的预测线，计算下面的结果，得到一个cls loss值，它是使用一个batch 中所有符合条件的actor算出来的。

![image](https://user-images.githubusercontent.com/95835767/145328947-4f269f23-5912-468d-a06d-1cfefda05b07.png)

K = 6 ，M= actor_num, $c_{m,k}$为对应actor和预测轨迹线的置信度分数，

![image](https://user-images.githubusercontent.com/95835767/145328954-08c0e3bc-8b7f-42bf-86c6-bbcc8b25bd39.png)

p为对应预测线的30个时刻的位置。

经后处理后，得到一个结果矩阵metric

**cls_loss**

num_cls 满足两个条件：mask1：(actor_num,1)最小最终距离小于阈值mask2：（actor_num,6）各个预测线的最终距离与最小最终距离差距大于阈值，num_cls取两个mask的乘积，也是最终聚酸cls_loss所用到的数据

**reg_loss**

num_reg:所有actor30个未来预测时刻有真值的数量加和

**pred:**所有batch的预测值，在一个epoch中，每个batch 后都append这个batch的**agent**预测数据（1，6，30，2），一个epoch结束后清空

gt_pred:所有数据的agent的真实值（1，30，2）

has_pred:agent在30个时刻是否有真实值（1.30）

当一个epoch完成之后

cls = cls_loss/num_cls

reg = reg_loss/num_reg

loss = cls+reg

err是agent6条预测线在30个时刻的预测误差（data_num,6,30）

ade fde是取最后一个时刻预测误差最小的那条预测线

ade是计算30个时刻的预测误差平均值，对所有数据的agent的所有30个预测点的误差取平均值，得到一个数

fde是所有数据的agent最后时刻的误差取平均值



ade1 fde1是所有数据都取六条预测线中置信度最高的那条

ade1 对所有数据的agent的所有30个预测点的误差取平均值，得到一个数

fde1是所有数据的agent最后时刻的误差取平均值

## Test

加载训练好的模型，根据测试数据计算测试结果，取output中第一个actor的6个预测结果，也就是agent的预测结果，与data["gt-preds"]作为真值。

### 评测指标

平均位移误差（ADE）：预测位置和真实位置之间的l2距离，对所有的步骤求平均值。

最终位移误差（FDE）：预测范围最后一步预测位置和真实位置之间的l2距离。

MinADE、MinFDE 取K=6个预测的最小ADE和最小FDE作为指标

MinADE、MinFDE：

将每一个csv文件对应的agent的6条预测线，取相应的ade和fde最小的那个ade和fde值，所有的csv文件对应的ade和fde值求平均值。

MR：miss rate指每一个csv文件对应场景下agent的minfde>2米，也就是最好预测线的最后时刻距离真实值大于2米的场景占所有场景的比率

DAC：6条预测线，每条预测线30个预测点全为可驾驶区域的预测线个数/预测线总数

