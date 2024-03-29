# 我最喜欢的统计量：霍夫丁的D

![插图](https://raw.githubusercontent.com/Dicklesworthstone/hoeffdings_d_explainer/main/hoeffd_illustration.webp)

假设你有两组数字序列，你想比较它们以测量它们之间的相关性或依赖程度。这实际上是一个非常通用的设置：这两个序列可以代表时间序列，这样你就会有一个包含三列和许多行的表格。第一列是时间戳（比如，按小时间隔），然后每个序列各有一列；第一列可以是该时间间隔内股票的平均价格，第二列可以是该时间间隔内交易的股票数量。或者，你也可以比较一个股票的百分比价格变化与另一个股票的变化。当然，它们不必是时间序列：你也可以只有两列（即，根本没有时间戳列）。第一列可以是30岁以上美国人的身高（以英寸为单位），第二列可以是同一个人的体重（以磅为单位）。或者，用一个更及时的例子，每一列可以代表来自特定大型语言模型（LLM）的某个英语句子字符串的嵌入向量。第一列可以是对于字符串“I love my 3 sons”（我爱我的三个儿子）的Mixtral 8x7B模型的嵌入向量，另一列可以是对于字符串“I cherish my 5 daughters”（我珍爱我的五个女儿）的同一个模型的嵌入向量。

在这些案例中，我们有两组数据序列，我们想要进行比较。问题在于，在最通用的设置中，我们可能事先没有任何线索关于这些关系的性质，或者甚至是否存在可以谈论的关系。如果两个序列完全独立，就像两个不同公平骰子的滚动记录呢？如果数据有点乱，包含一些极端的异常值条目，扭曲了你可能想要查看的最常见的测量类型，比如每列的平均值和方差呢？你现在可能会想，“等一下，解决这个问题的方法不就是查看相关性吗？”这确实是一个好主意，因为它是最常用的衡量两组数据集关联的方法。

为了澄清我们的术语，“相关性”通常指的是皮尔逊相关系数，其起源可以追溯到1800年代。这种相关性实际上只是数据之间的协方差的一个重新缩放，以提供一个不依赖于特定单位的度量。但是，协方差是什么呢？直观地，你首先单独观察每个序列并计算其平均值。然后看看该序列的单个数据点与该平均值的偏差（即，你只是从该序列的平均值中减去每个数据点）。然后通过将它们相乘来比较每个序列的这些度量。如果序列显示出相似的模式，即当第一个序列中的条目倾向于大于该第一个序列的平均值“同时”第二个序列也是如此，这表明它们是相关的，这些度量的乘积将会更高。（我把“同时”放在引号中，因为我们真正的意思是“在两个序列的第K个条目中”，因为这些序列不必是时间序列；谈论时间只是为了使其更易于理解）。如果序列实际上彼此没有任何关系（比如，它们只是简单记录公平骰子的滚动），那么这个乘积将接近零，因为当一个序列的条目在同一时间高于其平均值时，另一个序列的条目低于其平均值的可能性同样大。

皮尔逊相关性，通过我们重新缩放协方差以给出一个无量纲数，给你一个漂亮的、容易解释的结果：它总是在-1.0和1.0之间；如果它等于0.0，那么序列之间“没有关系”；如果是1.0，那么它们完全相关，并且步调一致。如果是-1.0，那就完全相反：一个上升时，另一个以同样的量下降。所以这听起来像是一个非常好的度量，对吗？确切的问题是什么呢？问题在于，当你使用皮尔逊相关性时，你隐含地寻找两个序列之间的一种特定类型的关系：线性关系。而且，在生活中你可能想要比较的许多事物甚至都不是模糊线性的。

让我们使用一些具体的例子来使事情更加具体化，使用我们之前提到的人的身高和体重的例子。我们预期它们之间存在大致的线性关系。当然，有一些非常矮但非常胖的人，也有一些超高但非常瘦的人，但平均来说，我们预期存在一个大致的线性关系，这样，如果你观察一个散点图，横轴表示身高，纵轴表示体重，每个点代表你样本中的一个人，如果从整个人群中无偏差地选取足够多的人，你应该能够通过拟合这些数据的线性模型来获得一个大致的模型。这个模型与实际数据点的详细比较程度被称为R平方；基本上是一个数据序列的方差有多少百分比被另一个数据序列解释了。

在这种情况下，皮尔逊相关性（Pearson’s correlation）工作得很好，因为它相对简单且容易计算（例如，你可以在Excel中轻松完成）且易于理解和解释，它已成为衡量关联或依赖性的“首选”方法。然而，还有很多其他情况，其中两组数据序列之间非常明显地存在可以理解的关联，但并不是像那样的线性关系。例如，想象一个成年人的体重和他们在50码短跑比赛中达到的最高速度，其中他们尽可能地跑得快。人们可以想象，非常瘦的人可能没有那么多快速收缩的腿部肌肉，因此可能会比较慢，然后随着体重的增加，平均最高速度首先增加。但显然，非常超重的人跑步会更慢，所以平均最高速度在某一点后开始下降，一旦体重达到非常严重的肥胖级别，速度会急剧下降。简而言之，这不是你能很好地拟合一条线的情况。而皮尔逊相关性的问题在于，它在这种情况下是不可靠的。

皮尔逊相关系数的另一个缺点是先前提到的：对异常值的敏感性。如果输入跑步者数据的人犯了错，漏掉了一个数字，或者在某人的体重上多加了几个零会怎样？即使你有一个包含一千人的数据集，如果其中一个人错误地被输入数据为重200万磅，那可能会非常戏剧性地搞砸你的所有测量。虽然这个特定的例子听起来可能有些牵强，但这种异常数据问题在实践中非常常见，特别是在处理大型数据集时。

为了处理异常值问题，已经提出了对皮尔逊相关系数的小幅修改。例如，斯皮尔曼等级相关（Spearman's Rho）本质上就是皮尔逊相关系数，但你首先需要将每个数据点替换为其在各自序列中的排名。例如，如果上一个例子中最重的人重200万磅，那个超高值将被替换为1000（因为那个数据集中有1000人，而200万磅的人将是那个集合中最重的）。这提供了一种测量方法，总体上与皮尔逊的工作方式相似，但对异常值具有鲁棒性。

还有一种对斯皮尔曼方法的进一步细化叫做肯德尔秩相关系数（Kendall's Tau）。它也用它们的排名替换数据点，但不是像斯皮尔曼等级相关那样将所有数据点一起考虑，而是取数据点的个别对。例如，你可能取每个序列的第三个数据点，并通过询问它们在各自序列中的排名是否和谐（两个排名都高于或低于另一对）或冲突（一个序列中的排名较高但在另一个序列中较低——这些被称为“一致”和“不一致”的对）。你重复这个过程对所有数据点进行操作，本质上就像对所有可能的对进行一次意见一致与否的计数。这里有一个微妙的点是如何处理并列的情况，即每个序列的第K个条目在其各自的序列中有相同的排名。肯德尔秩相关系数考虑了并列的数量，以便我们可以有效地“重新标定”或标准化关联度量。直观地，如果我们有两个向量X和Y，每个包含一百万个数值观察结果，而这些中除了5个外全部在X和Y之间逐元素相同，这与X和Y的长度仅为10且有5个不同条目的情况是非常不同的。

所以，肯德尔系数有效地处理了异常值，并且与皮尔逊相关系数不同，它不假设序列之间存在线性关系。然而，当涉及到可靠地检测两个序列之间可能存在的某些类型的关系时，它仍然存在严重问题——特别是在揭示复杂的非线性关系方面。肯德尔系数的一个缺点，以及许多其他基于等级的方法的缺点是，当你处理更复杂的关联模式时，它们会失效，比如周期性的关系或涉及多个变量以非单调方式相互作用的关系。这是因为肯德尔系数关注于对之间的一致性和不一致性，这只适用于单调关系，其中变量相对于彼此以一种一致的方向移动。

问题实际上比关系是否单调更加复杂。要全面理解这个问题的最佳方式是想象一些奇怪的散点图，这些散点图对于视觉上观察它们的人来说会立即提示出某些明显和清晰的关于两个序列之间关系的描述，但是迄今为止我们讨论的测量方法对此几乎无话可说。例如，假设散点图看起来像一个环形，或者“X”形状。如果你想直观感受我在这里说的内容，[这个页面](https://www.wolfram.com/mathematica/new-in-9/enhanced-probability-and-statistics/use-hoeffdings-d-to-quantify-and-test-non-monotoni.html)有一些非常好的例子，并展示了霍夫丁的D如何捕捉到通常被忽视的各种奇怪形状。如果你只使用肯德尔系数或皮尔逊相关系数来测量这些形状，它们可能看起来几乎没有关联，但是如果你问那些通过视觉进行分析的人，给定一个点的水平位置（或相反），他们认为这个点在垂直轴上应该在哪里，他们可能有一个非常好的想法（尽管这可能被表述为“环形结构的顶部或底部——但很可能是其中一个。”）这种关系非常远离随机，然而，由于它违反了一些基本假设（即，关系可以使用一个变量的函数来描述，通过你可能从微积分课程记得的“垂直线”测试），它在很大程度上对这些更简单的关联度量来说是“看不见的”。

如果你从大型语言模型（LLM）和嵌入向量相似度的角度来探讨这个话题，用以量化语言文本字符串之间的“语义相似性”，你可能会问：“但是余弦相似度呢？那不是几乎所有向量数据库和RAG（Retrieval-Augmented Generation）管道使用的黄金标准吗？”。这是个好问题！余弦相似度确实非常有用，而且重要的是，它能够快速有效地计算数百万甚至数十亿个向量。从直观层面上看，余弦相似度通过将每个序列视为N维空间中的一个点来工作。所以，如果你的每个嵌入向量是2048个数字长，你有2个这样的向量，你可以将它们视为构成2048维空间中的两个独立点。然后，你比较这些不同维度中，每个点相对于原点所形成的角度。如果这些点跨越的向量与原点形成类似的角度，那么在某种意义上它们“指向这个高维空间中相同的大致方向”，因此它们彼此“相似”。这种概念化方式的问题在于，一旦我们的空间直觉超过3个维度，就开始崩溃，到了2048维，事情变得非常奇怪和非直观。例如，在如此高维空间中，球体或立方体的一般化几乎所有的体积都包含在其表面附近，而对于3D球体或立方体则恰恰相反。

尽管如此，当你需要在海量数据中找到针的大概位置时，余弦相似度极为方便。如果你的向量数据库中有数百万个句子的嵌入向量，来自数千本书，而你想找到与句子“人们普遍认为，古代最伟大的数学家是阿基米德。”相似的句子，那么它对于快速排除这数百万句子中99.999%与这个非常具体的想法无关的句子非常有效，因为大多数句子对应的嵌入向量指向嵌入空间中非常不同的位置。但是，当你已经过滤掉几乎所有存储的向量，只剩下20个“最相似”的句子时，现在你想要对这些句子进行排名，以便首先显示最相关的句子怎么办？我认为，在这里余弦相似度可能是一种钝器，可能以各种方式被扭曲。但更好的例子可能是，当你的向量数据库中根本就没有明显相关的句子，以至于通过余弦相似度找到的20个“最相似”的句子看起来都不特别相关。在这种情况下，我们可能想要有另一种关联或依赖性的度量，我们可以将其应用于通过余弦相似度找到的顶部1%或0.1%相关向量，以获得这些的排名顺序。

因此，你现在可以看到我们为什么可能想要找到一个更强大、更通用的关联或依赖度量，一个不对我们两个数据序列之间可能的关系性质做出假设的度量，不要求关系是1对1函数或单调的，可以轻松容忍错误的异常数据点而不完全崩溃的度量。我的观点是，霍夫丁的D是迄今为止发现的最佳度量。它是由波兰数学家Wassily Hoeffding在他1948年的论文《一种非参数的独立性检验》中首次介绍的。在这篇12页的论文中，Hoeffding定义了他所称的**D**（代表依赖性）。如果你有数学背景，你可能会想读他的原始论文，但我怀疑大多数人会发现它很难理解和不直观。然而，这并不是一个根本难以理解的概念，如果以最简单和最直观的方式呈现，你是可以理解的，这就是我现在尝试要做的。

像肯德尔的Tau一样，计算霍夫丁的D开始的方式类似：你首先用该值在各自序列中的排名替换你两个序列中的每个值；如果多个值完全相等，因此你有“平级”，那么你取等值的排名平均值。因此，如果一个值4.2使得一个特定数据点在你的一个序列中1000个数据点中的排名为252，但实际上在序列中有4个这样值为4.2的点，那么这些点每个都会收到平均排名(252 + 253 + 254 + 255)/4 = 253.5。然后你还需要查看每个序列中的点对，看看有多少是“一致的”或“不一致的”，这也类似于肯德尔的Tau的工作方式。但然后过程发生了分歧：在对数据进行排名并考虑对之间的一致性和不一致性之后，霍夫丁的D引入了一种独特的方法来量化两个序列之间的依赖性。它计算一个基于观察到的“联合分布”排名的差异和如果两个序列是独立的所期望的差异的统计量。

让我们先退一步，解释我们所说的“联合分布”与我们两个序列中的每个的个别或“边际”分布相比意味着什么。在霍夫丁的D的上下文中，当我们谈论“排名的联合分布”时，我们指的是两个序列中的排名如何在所有数据点对中结合或相互关联。想象绘制一个图表，其中x轴代表一个序列中的排名，y轴代表另一个序列中的排名。这个图表上的每个点代表一对排名——每个序列中的一个。这些点在图表上形成的模式反映了它们的联合分布：它向我们展示了一个序列中的排名如何与另一个序列中的排名相关联。

另一方面，“边际分布”涉及单个序列内的排名，被单独考虑。如果我们只看上述图表的一个轴（x轴或y轴），并忽略另一个轴，沿着该轴的点的分布代表了该序列的边际分布。它告诉我们该序列内部排名的分布或扩散情况，而不考虑这些排名如何与另一序列的排名配对。

理解联合分布和边际分布之间的区别对于掌握霍夫丁的D值是至关重要的。该度量本质上评估观察到的排名的联合分布是否偏离了如果序列是独立的预期情况。在独立性下，联合分布仅仅是边际分布的乘积——意味着，跨序列的排名配对是随机的，没有可辨识的模式。然而，如果序列之间存在依赖性，观察到的联合分布将与这种边际的乘积不同，表明一个序列中的排名与另一个序列中的排名系统性地相关。霍夫丁的D值量化了这种差异，提供了序列间依赖性的统计度量。

为了做到这一点，霍夫丁的D值考虑了所有可能的“四元组”排名对。也就是说，对于任意四个不同的数据点，它检查一对排名与序列中另一对排名的一致性。这涉及将每对数据点与所有其他对比较，这一过程比肯德尔的Tau更全面，后者只查看一致和不一致的对。霍夫丁的D值的本质在于对数据点的联合排名的评估。它计算了这些比较得出的某些项的总和，这些项反映了所有对和四元组的一致性和不一致性程度。这些项考虑了一个序列中的数据点在与另一序列比较时，排名在另一个点上方和下方的次数，对于平局进行了调整。

霍夫丁的D的最终计算涉及一个公式，该公式在考虑到数据点的总数和在独立性假设下的期望值时，对这个总和进行规范化。结果是一个范围从-0.5到1的度量，其中数字越高，两个序列之间的依赖性就越强。你可以想象，对两个一定长度的序列计算霍夫丁的D（比如，每个序列有5,000个数字）涉及到大量的个别比较和计算——远远超过用来得出肯德尔的Tau，更不用说斯皮尔曼的Rho或皮尔逊相关系数的计算了，因为我们考虑的是整个序列，不仅仅是个别对，而且还要深入到个别对的水平。

对于每个包含5,000个数字的两个序列，霍夫丁的D不仅仅是一次比较每个点与其他每个点（这已经相当可观）；它检查了来自合并数据集的所有可能的四元组点之间的关系。为了把这个问题放入透视中，如果你比较单一序列中5,000个点的每对点，你会进行大约1250万次比较（因为5,000选2大约是1250万）。但霍夫丁的D需要比较四元组。5,000个序列中唯一四元组的数量由组合公式'n choose 4'给出，对于5,000来说大约是62亿个四元组。对于这些四元组中的每一个，霍夫丁的D涉及多个比较和计算，以评估它们在整个数据集上下文中的一致性和不一致性。

这种比较数量的指数增长强调了霍夫丁的D在计算上要求显著更高的原因。这不仅仅是规模问题；它是霍夫丁D进行的分析的深度和广度，以捕捉两个序列之间的复杂依赖性。这种全面的方法允许霍夫丁的D检测到简单度量可能错过的微妙和复杂的关联，但这也意味着其计算可能是资源密集型的，特别是对于大型数据集。但我认为，在一个廉价和快速计算的时代，现在是时候开始利用霍夫丁D的许多优势了。除了允许我们找到两个序列之间任何类型的关系而不做任何关于它们分布的假设，以及对异常值的鲁棒性之外，霍夫丁的D还有其他一些好处：它是对称的，这样hoeffd(X, Y)和hoeffd(Y, X)是相同的，它总是有界的（结果永远不会小于-0.5或大于1.0），而当序列之一是常数时，霍夫丁的D趋向于零。这不适用于一些其他强大的关联度量，如*互信息*（我们在这里没有讨论）。

现在我们已经给出了基本概述，让我们深入了解这一切实际是如何完成的细节。首先我会用文字加上一个公式给你解释（这是唯一一个尝试避免使用公式可能会让事情看起来比它们已经是的更加混乱和复杂的地方！）。如果这听起来非常混乱，几乎没有任何意义，也不用担心。看看这些想法通常是如何呈现的很有用（相信我，原始的1948年的论文甚至更难以解析！）。然后我们将一点一点地分解它，试图给出每一部分的直觉。最后我们将看看一个实际的（但慢的）Hoeffding的D在Python中的实现，这个实现有广泛的注释。

**逐步分解：**

1. **排名和成对比较**：
    - 最初，两个序列中的每个数据点都被替换为其在各自序列中的排名，对于平局的值，通过分配平均排名来考虑平局。
    - 比较涉及查看这些排名序列中所有可能的对，以确定它们是一致的（两个排名一起增加或减少）还是不一致的（一个排名增加而另一个减少）。
    
2. **四元组比较**：
    - 超越成对比较，Hoeffding的D评估所有可能的四元组数据点。对于每个四元组，该度量评估一个排名对内的顺序是否与另一个排名对内的顺序一致。这一步对于捕捉超出简单成对关联的更复杂依赖性至关重要。我们将这些排名存储在一个我们称为**Q**的数组中，它为每个数据点持有一个值，反映其与其他点在排名一致性和不一致性方面的关系。对于每个数据点，**Q**累积计算有多少对（与另一个数据点一起考虑时为四元组）显示一致（一致的）或不一致（不一致的）排名行为。这一步对于捕捉成对比较可能错过的复杂依赖性至关重要。
    
3. **求和**：
    - Hoeffding的D计算的核心涉及对所有四元组的一致性和不一致性评估中得到的某些项进行求和。这个和反映了观察到的排名联合分布与如果序列是独立的预期分布的偏差程度。
    
4. **标准化**：
    - 最后的计算涉及对前一步骤获得的和进行标准化。这种标准化考虑了数据点总数（**N**）并调整了独立假设下的预期值。这种标准化的目的是缩放Hoeffding的D统计量，使其可以跨不同样本大小和分布进行比较。
    - 标准化公式为：
    ```math
     D = \frac{30 \times ((N-2)(N-3)D_1 + D_2 - 2(N-2)D_3)}{N(N-1)(N-2)(N-3)(N-4)}
    ```
    - 这里，**D_1**、**D_2**和**D_3**是涉及排名及其一致性/不一致性评估的组合的中间和。具体来说：
        - **D_1**与所有四元组的排名差异乘积的总和

相关，反映了总体的一致性/不一致性。
        - **D_2**调整了每个序列内的个体变异性。
        - **D_3**考虑了序列之间的相互作用。

既然这一切看起来还是相当复杂，让我们来详细说明什么是四元组以及**Q**和**D_1**、**D_2**和**D_3**元素。

**Q是什么及其目的？**

- **Q**代表每个数据点的加权计数，考虑到在两个维度（例如，身高和体重）中有多少其他点的排名低于自己的排名，并对平局进行调整。
- 它对于捕捉数据点之间的一致性和不一致性程度至关重要，超越了简单的成对比较。
- **Q**的计算包括对平局的调整，确保平局排名适当地贡献到整体测量中。
- 通过考虑相对于给定点的所有其他点并纳入平局的调整，**Q**值提供了每个点在排名联合分布中位置的细腻视角，这是评估两个序列之间依赖性的基础。

**对D_1、D_2、D_3的澄清**：

这些中间和在计算霍夫丁的D时扮演着不同的角色：

  - **D_1** 反映了在独立性预期下调整后的所有四元数据点之间的一致性/不一致性的聚合。它是衡量实际数据与序列之间没有关系时预期情况偏离程度的指标。
      - **D_1的直觉理解**：将**D_1**视为量化两个序列之间的协调变化程度，超出了随机机会所产生的范围。它有效地捕捉了排名顺序的相互影响程度，评估配对观察值是否比偶然情况下预期的更经常地以一致的模式一起移动。
      - **D_1的非常简单的直觉理解**：想象你正在比较两位舞者在各种表演中舞步的同步性。**D_1**代表他们的动作比单纯偶然情况下预期的更同步（或不同步）。它通过量化他们的协调（或不协调）努力来捕捉他们伙伴关系的本质。
  
  - **D_2** 代表每个序列内排名方差的乘积，提供了一个独立于任何序列间关系的变异性基线测量。它有助于从它们的相互依赖中分离出序列内部变异性的影响。
      - **D_2的直觉理解**：**D_2** 评估了每个序列自身内在的变异性或分散性。它类似于评估每个序列中数据点的扩散，以了解每个序列独立变化的程度。这有助于区分由序列本身的固有变异性和由它们相互作用产生的依赖性。
      - **D_2的非常简单的直觉理解**：考虑**D_2**为评估每位舞者在表演中展示的舞蹈风格范围，独立于他们的伴侣。它衡量每位舞者在他们的表演中是多样化还是一致性。通过理解这种个体变异性，我们可以更好地辨别他们的同步性（由**D_1**捕捉）在多大程度上是由于他们的互动而不是他们的个体倾向。
  
  - **D_3** 作为一个互动项，融合了**D_1**中的见解与**D_2**捕捉的序列内部变异性。它通过考虑个别排名一致性如何贡献于整体依赖性，考虑每个序列的内部排名结构，来微调测量。
      - **D_3的直觉理解**：**D_3**通过考虑每个序列内部的个体变异性（由**D_2**捕捉）如何影响观察到的一致性/不一致性（**D_1**），来调整测量。它关于理解每个序列的内部结构如何影响它们的共同行为，通过考虑个体变异性在它们观察到的依赖性中的作用来细化它们关系的评估。
      - **D_3的非常简单的直觉理解**：**D_3**就像是

考察每位舞者的个体风格和一致性（**D_2**）如何影响他们的联合表演（**D_1**）。如果一位舞者有广泛的风格范围，这种多样性如何影响他们与伴侣的同步性？**D_3**评估每位舞者的变异性对他们协调努力的影响，提供了他们伙伴关系的细致视角。

**霍夫丁's D值的最终计算**：

- 霍夫丁's D的最终公式结合了**D_1**、**D_2**和**D_3**，以及归一化因子，产生一个范围在-0.5到1之间的统计量。
    - **最终计算的直观理解**：最终的霍夫丁's D值是一个综合性的度量，整合了观察到的一致性/不一致性、每个序列内部的固有变异性，以及这些因素之间的相互作用。归一化确保了该度量针对样本大小进行了适当的缩放，使其成为序列间关系强度和方向的稳健指标。它将复杂的相互依赖性和个体变异性蒸馏成一个反映整体关系的单一指标，不仅考虑了关系的存在，还考虑了其性质和相对于独立性预期的强度。
    - **非常简单的最终计算直观理解**：最终的霍夫丁's D分数类似于舞蹈比赛中的最终得分，评估两位舞者之间的整体和谐度。这个最终得分将他们表演的所有方面——个人风格、一致性和相互作用——蒸馏成一个单一的、可解释的度量，反映了他们的舞伴合作。

尽管你现在对如何计算霍夫丁的D有了更好的理解，但事情可能仍然显得抽象，如何在真实的编程语言中实际实施整个过程可能还不太清楚。所以，我们现在要用Python来做这件事；为了保持简短，我们将利用Numpy和Scipy库：Numpy用于有效地处理数组（包括方便的索引），Scipy的“rankdata”函数用于高效地计算等级，通过平均处理我们需要的方式来处理平局。

为了使事情变得非常具体，我们将使用实际的10个数据点：它们将是（`height_in_inches`，`weight_in_pounds`）形式的数据对：

```python
X = np.array([55, 62, 68, 70, 72, 65, 67, 78, 78, 78])  # Heights in inches
Y = np.array([125, 145, 160, 156, 190, 150, 165, 250, 250, 250])  # Weights in lbs
```

为了保持回答的简洁性，我没有展开霍夫丁D的详细计算步骤。如果你需要完整的实现代码，包括如何计算**D_1**、**D_2**和**D_3**，以及如何结合它们，请让我知道，我可以提供具体的实现细节。

我们程序对该输入运行的结果如下：

- 身高的排名（X）：[1. 2. 5. 6. 7. 3. 4. 9. 9. 9.]
- 体重的排名（Y）：[1. 2. 5. 4. 7. 3. 6. 9. 9. 9.]
- Q 值：[1.  2.  4.  4.  7.  3.  4.  8.5 8.5 8.5]
- 数据的霍夫丁's D：0.4107142857142857

信不信由你，我们实际上可以仅用大约15行Python代码实现全部霍夫丁的D！然而，我们将添加大量注释并显示一些中间值，因为重点是尽可能清晰地说明。所以，包括空白在内，我们最终将得到大约75行代码——考虑到代码的功能，这还是不多的！

如果你想实际运行代码，你可以在这里查看Google Colab笔记本：

https://colab.research.google.com/drive/1P5hFJKVS2D6wT9nOdXs_Gw34QVzY0p8e?usp=sharing


这是完整的代码：

```python
    import numpy as np
    import scipy.stats
    from scipy.stats import rankdata
            
    def hoeffd_example():
        # 新的数据集包含10个数据点，表示（身高（英寸），体重（磅））的配对，最后3个是并列的：
        X = np.array([55, 62, 68, 70, 72, 65, 67, 78, 78, 78])  # 身高（英寸）
        Y = np.array([125, 145, 160, 156, 190, 150, 165, 250, 250, 250])  # 体重（磅）
        # 'average'排名方法为所有并列值分配排名的平均值。
        R = rankdata(X, method='average')  # 身高的排名
        S = rankdata(Y, method='average')  # 体重的排名
        # 打印排名以便可视化。在两个排名的最后三个条目中并列情况显而易见。
        print(f"身高（X）的排名: {R}")
        print(f"体重（Y）的排名: {S}")
        N = len(X)  # 数据点的总数
        # Q是一个数组，将为每个数据点保存一个特殊的和，这对于计算霍夫丁的D值至关重要。
        Q = np.zeros(N)
        # 循环遍历每个数据点以计算其Q值。
        for i in range(N):
            # 对于每个数据点'i'，计算有多少点的身高和体重排名都低（一致对）。
            Q[i] = 1 + sum(np.logical_and(R < R[i], S < S[i]))
            
            # 为并列调整Q[i]：当两个排名都相等时，它部分地（1/4）贡献给Q[i]值。
            # “- 1”是因为不在自己的比较中包括该点。
            Q[i] += (1/4) * (sum(np.logical_and(R == R[i], S == S[i])) - 1)
            
            # 当身高排名并列但体重排名更低时，它贡献一半（1/2）给Q[i]值。
            Q[i] += (1/2) * sum(np.logical_and(R == R[i], S < S[i]))
            
            # 同样，当体重排名并列但身高排名更低时，它也贡献一半（1/2）。
            Q[i] += (1/2) * sum(np.logical_and(R < R[i], S == S[i]))
        # 打印每个数据点的Q值，表示被认为是“更低”或“相等”的点的加权计数。
        print(f"Q值: {Q}")
        # 计算霍夫丁的D公式所需的中间和：
        # D1：这个和利用了之前计算的Q值。每个Q值封装了关于数据点的排名如何与两个序列中的其他排名相关的信息，包括一致性和平局的调整。
        # 对于每个数据点，项(Q - 1) * (Q - 2)量化了这个点的排名在多大程度上与其他点一致，经过了独立性下预期一致性的调整。
        # 跨所有数据点求和（D1）聚合了整个数据集的一致性信息。
        D1 = sum((Q - 1) * (Q - 2))
        # D2：这个和涉及每个序列的排名差异的乘积，经过平局调整。对于每个数据点，项(R - 1) * (R - 2) * (S - 1) * (S - 2)
        # 捕捉了每个序列内排名方差之间的相互作用，提供了一个衡量联合排名分布与独立性下预期的偏离程度的度量，
        # 这种偏离仅由排名的变异性引起，而不考虑它们的配对。
        # 跨所有数据点求和（D2）给出了这种偏离的全球评估。
        D2 = sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
        # D3：这个和代表了一个结合了Q值和排名差异见解的相互作用项。
        # 对于每个数据点，项(R - 2) * (S - 2) * (Q - 1)考虑了排名方差与Q值一起，捕捉了个别数据点的排名一致性/不一致性如何
        # 贡献于整体的依赖度量，经过了独立性下预期值的调整。求和这些项（D3）将这些个别贡献整合成了数据集的一个综合性相互作用项。
        D3 = sum((R - 2) * (S - 2) * (Q - 1))
        # 最终计算霍夫丁的D值整合了D1、D2和D3，以及考虑样本大小（N）的归一化因子。
        # 归一化确保了霍夫丁的D值被适当地缩放，允许对不同大小的数据集进行有意义的比较。这个公式以一种平衡一致性、不一致性和排名方差的贡献的方式
        # 结合了这些和和归一化因子，结果是一个鲁棒地衡量两个序列之间关联程度的统计量。
        # 返回计算得到的霍夫丁D值。
        D = 30 * ((N - 2) * (N - 3) * D1 + D2 - 2 * (N - 2) * D3) / (N * (N - 1) * (N - 2) * (N - 3) * (N - 4))
        # 返回计算出的霍夫丁的D值。
        return D
    # 计算并显示数据集的霍夫丁D值
    hoeffd_d = hoeffd_example()
    print(f"数据的霍夫丁D值: {hoeffd_d}")

```

现在，我提到了这个实现虽然是正确的，但相当慢且效率低下。考虑到我们使用Numpy和Scipy进行“重型计算”，而这些实际上是用高效的编译型C++实现的，这可能会让人感到惊讶。事实证明，鉴于涉及的大量操作和比较，函数的开销如此之大，以至于即使在快速的机器上，一旦数据点超过1,000个，代码真的开始变得非常缓慢，而处理5,000个数据点将需要极长的时间。

幸运的是，我用 Rust 写了一个更高效的版本，可以轻松地作为 Python 库使用。你只需执行以下操作即可使用：

`pip install fast_vector_similarity`

你可以在这里查看代码：

https://github.com/Dicklesworthstone/fast_vector_similarity

还可以在这里查看关于该库在 Hacker News 上的讨论：

https://news.ycombinator.com/item?id=37234887