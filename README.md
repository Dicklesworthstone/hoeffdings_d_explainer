# My Favorite Statistical Measure: Hoeffding’s D
![Illustration](https://raw.githubusercontent.com/Dicklesworthstone/hoeffdings_d_explainer/main/hoeffd_illustration.webp)

Suppose you have two sequences of numbers that you want to compare so you can measure to what extent they are related or dependent on each other. It’s really a quite general setting: the two sequences could represent time series, so that you have a table with three columns and a bunch of rows. The first column would be a timestamp (say, at hourly intervals), and then one column for each sequence; the first could, for example, be the average price of a stock during that interval, and the second could be the volume of shares traded traded during that interval. Or you could compare the percentage price change of one stock to that of another. Of course, they don't need to be time series at all: you could also have just two columns (i.e., no timestamp column at all). The first could be the height of an American over age 30 in inches, and the second could be the weight of that same person in pounds. Or, to use a more timely example, each column could represent an embedding vector of some English language sentence string from a particular LLM. The first column could be the embedding vector from the Mixtral 8x7B model for the string “I love my 3 sons” and the other could be the embedding vector from the same model for the string “I cherish my 5 daughters.”

In each of these cases, we have two sequences of data that we want to compare. The problem is that, in the most general setting, we might have no clue a priori what the nature of the relationship might be, or if there even *is* a relationship to speak of. What if the two sequences are totally independent, like recordings of the rolls of two different fair dice? What if the data is a little screwed up and contains some extreme outlier entries that distort the most common kinds of measures you might want to look at, such as the mean and variance of each column separately? You might think to yourself now, “Wait a second, isn’t the answer to this to just look at the correlation?” And that is certainly a good idea to check, since it’s the most commonly used measure of association between two data sets. 

To clarify our terms, what “correlation” generally refers to is Pearson’s correlation coefficient, which dates back to the 1800s. This correlation is really just a rescaled covariance between the data to give a measure that doesn’t depend on the particular units used. But then, what is covariance? Intuitively, you first look at each of the sequences individually and compute its mean value. Then look at how the individual data points for that sequence deviate from that mean (i.e., you just subtract each data point from the mean value for that sequence). Then you compare these measures for each sequence by multiplying them together. If the sequences display a similar pattern in that, when an entry in the first sequence tends to be larger than the mean for that first sequence “at the same time” as this holds for the second sequence, then this suggests they are related, and the product of these measure will be higher. (I put “at the same time” in quotes because what we really mean is “in the Kth entry for both sequences,” since the sequences don’t have to be time series; talking about time just makes it easier to understand). If the sequences truly have nothing to do with each other (say, like the case where they are simply recording the rolls of fair dice), then this product will be close to zero, because it’s just as likely for one of the sequences to have an entry that’s above its mean value at the same time the other sequence has an entry that’s below its mean value.

Pearson correlation, where we rescale the covariance to give a dimensionless number, gives you a nice, easily interpretable result: it’s always between -1.0 and 1.0; if it equals 0.0, then there’s “no relationship” between the sequences; if it’s 1.0, then they are perfectly correlated with each other and move in lockstep. If it’s -1.0, then it’s the exact opposite: when one goes up, the other goes down by the same amount. So this sounds like a pretty good measure, right? What’s the problem exactly? Well the problem is that, when you use Pearson correlation, you are implicitly looking for a particular kind of relationship between the two sequences: a linear relationship. And there are lots and lots of things in life that you might want to compare that aren’t even vaguely linear in nature.

Let’s use some concrete examples to make things more tangible and use the example we mentioned above of the height and weight of people. We would expect there to be a roughly linear relationship there. Sure, there are some very short but very fat people, and there are some super tall but very skinny people, but on average, we would expect there to be a roughly linear relationship so that, if you looked at a scatter plot showing the height on the X-axis and the weight on the Y-axis, where each dot represents a person in your sample, with enough people taken from the population as a whole without bias, you should be able to get a rough model by fitting a line to this data. The extent to which this model is accurate when compared to the actual data points in detail is called the R-squared; it’s basically what percentage of the variance of one data sequence is explained by the other data sequence.

In this case, Pearson’s correlation works well, and since it’s relatively simple and quick to calculate it (you can easily do it in Excel, for example) and simple to understand and interpret, it has become the “go to” measure of association or dependency. However, there are many other situations one can think of where there is very clearly an understandable association between two sequences of data that is not linear like that. For example, think about the weight of an adult and their top speed attained in a 50 yard dash race where they ran as fast as possible. Once can imagine that very skinny people might not have as much fast twitch leg muscle and thus might be on the slower side, and then as the weight increases at first, the average top speed increases. But then obviously very overweight people are going to be slower at running, so the average top speed would start to fall at some point, and then plunge quite rapidly once the weight gets to the very morbidly obese level. In short, it’s not something that you’d be able to fit a line to very well. And the problem with Pearson’s correlation is that it’s unreliable in a situation like this. 

Another drawback of Pearson’s correlation was referenced earlier: sensitivity to outliers. What happens if the person entering the data about the runners messed up and left out a number? Or added a few zeros to the weight of one of the people? Even if you had a data set of a thousand people, if one of the people is erroneously entered into the data as weighing 2 million pounds, that could very dramatically screw up all your measurements. And while this particular example may sound far-fetched, that sort of outlier bad data problem is very common in practice, particularly when dealing with large data sets.

For dealing with the issue of outliers, small modifications to Pearson’s correlation have been proposed. For example, Spearman’s Rho is essentially just Pearson’s correlation, but where you first replace each data point with its rank within its respective sequence. So for example, if the heaviest person in the previous example weighed 2 million pounds, that super high value would be replaced with 1,000 (since there were 1,000 people in that data set and the 2 million pound person would be the heaviest in that set). This gives a measure that works in a similar way overall to Pearson’s but is robust to outliers. 

There is another further refinement to Spearman’s approach called Kendall’s Tau. It also replaces data points with their ranks, but instead of looking at all of them together the way Spearman’s Rho does, it takes individual pairs of data points. For example, you might take the 3rd data point from each sequence and compares them to each other by asking if their ranks within their respective sequences are in harmony (both ranks higher or lower than another pair) or in conflict (one rank higher in one sequence but lower in the other— these are referred to as “concordant” and “discordant” pairs). You repeat this process for all the data points, and it’s essentially like taking a tally of agreements versus disagreements among all possible pairs. One subtle point here is how to handle the case of ties, where the Kth entry of each sequence has the same rank within its respective sequence. Kendall’s Tau takes the number of ties into account so that we can effectively “rescale” or normalize the resulting measure of association. Intuitively, if we had two vectors X and Y that each comprised one million numerical observations, and all but 5 of these were the same element-wise between X and Y, that's a very different situation than where X and Y are only of length 10 with 5 differing entries.

So Kendall's Tau effectively deals with outliers, and unlike Pearson's correlation, it doesn't assume a linear relationship between the sequences exists. However, it still has serious problems when it comes to reliably detecting the existence of certain kinds of relationships that might exist between the two sequences— especially in uncovering complex, non-linear relationships. One of the drawbacks of Kendall's Tau, and with many other rank-based methods, is that they break down when you are dealing with more intricate patterns of association, such as relationships that are cyclical or involve multiple variables interacting in a non-monotonic fashion. That’s because Kendall's Tau focuses on the concordance and discordance among pairs, which only works well for monotonic relationships, where the variables move in one consistent direction relative to each other. 

The issue is actually more involved than just whether the relationship is monotonic or not. The best way to appreciate the problem in all its generality is to think of some weird looking scatter plots that would instantly be suggestive to a person looking at them visually, in that there would be some obvious and clear thing you could say about the relationship between the two sequences, but where the measures we discussed so far would have very little to say. For example, suppose the scatter plot looked like a ring, or an “X” shape. If you want to get a visual sense of what I'm talking about here, [this page](https://www.wolfram.com/mathematica/new-in-9/enhanced-probability-and-statistics/use-hoeffdings-d-to-quantify-and-test-non-monotoni.html) has some very good examples and shows how Hoeffding's D can catch all sorts of weird shapes that are normally missed. Both of those might appear to have close to zero association if you only measured them using Kendall’s Tau or Pearson correlation, but if you asked the person doing things visually where they thought a point should go on the vertical axis given the horizontal position of the point (or vice versa), they might have a very good idea (although this might be couched as something like “either the top part of the ring structure here, or the bottom part of the ring structure here— but likely one or the other.”) That sort of relationship is very far from random, and yet it is largely “invisible” to these simpler measures of association because it violates some basic assumptions involved (i.e., that the relationship can be described using a function of one variable that passes the “vertical line” test that you might remember from a calculus course).

If you're approaching this topic from the standpoint of LLMs and embedding vector similarity, used to quantify “semantic similarity” between strings of language text, you might be asking “But what about cosine similarity? Isn’t that the gold standard used by nearly all vector databases and RAG pipelines?”. Good question! Cosine similarity is indeed very useful and, importantly, quick and efficient to compute across millions or even billions of vectors. At an intuitive level, cosine similarity works by thinking of each sequence as being a point in N-dimensional space. So if each of your embedding vectors is 2048 numbers long, and you had 2 of these vectors, you could think of them as constituting two individual points in 2048-dimensional space. You then compare the angle that each of these points makes relative to the origin across these various dimensions. If the points span vectors that make similar angles to the origin, then they in a sense “point in the same general direction” in this high dimensional space, and thus they are “similar” to each other. The problem with this way of conceptualizing it is that our intuitions about space break down once we get above 3 dimensions, and by the time you get to 2048 dimensions, things get very weird and unintuitive. For example, the generalization of a sphere or cube in such a high dimensional space would have nearly all of its volume contained near its surface, whereas the opposite is true for a 3D sphere or cube. 

That being said, cosine similarity is extremely handy for finding the approximate location of a needle in a haystack. If you have embedding vectors for millions of sentences from thousands of books in a vector database, and you want to find similar sentences to the sentence “The greatest mathematician of antiquity is generally thought to be Archimedes,” then it’s very effective for quickly eliminating the 99.999% of these millions of sentences that have nothing to do with this very specific thought, because most of those sentences will correspond to embedding vectors that point to very different locations in the embedding space. But what about after you’ve filtered out nearly all of the stored vectors and you are left with the 20 “most similar” sentences, and you now want to rank order these so the most relevant one is shown first? I would posit that this is where cosine similarity can be something of a blunt instrument, and can be distorted in various ways. But a better example might be the case where there simply is no obviously relevant sentence in your vector database, so that none of the top 20 “most similar” sentences found via cosine similarity look particularly relevant. In this case, we might want to have another measure of association or dependency that we can apply to the top 1% or 0.1% relevant vectors found via cosine similarity to get a rank ordering of these.

So now you can see why we might want to find a more powerful, general measure of association or dependence, one that doesn’t make assumptions about the nature of the possible relationship between our two data sequences, which doesn’t require that the relationship be a 1-to-1 function or monotonic, which can easily tolerate an erroneous outlier data point without breaking down completely. My claim is that Hoeffding’s D is the best measure yet discovered for this purpose. It was first introduced by the Polish mathematician Wassily Hoeffding in his 1948 paper entitled “A Non-Parametric Test of Independence”. In this 12-page paper, Hoeffding defines what he calls **D** (for dependency). If you have mathematical training, you might want to read his original paper, but I suspect most people would find it quite hard to understand and unintuitive. And yet it’s not such a fundamentally hard concept that you couldn’t understand it if it is presented in the simplest and most intuitive manner, which is what I will try to do now.

Like Kendall’s Tau, computing Hoeffding’s D starts out in a similar way: you first replace each value in your two sequences with the rank of that value within the respective sequence; if multiple values are exactly equal and thus you have “ties,” then you take the average of the ranks for the equal values. Thus if a value of 4.2 would make a particular data point have rank 252 out of 1000 data points in one of your sequences, but there are in fact 4 such points in the sequence with the exact value of 4.2, then each of these points would receive the average rank of (252 + 253 + 254+ 255)/4 = 253.5. You then also look at pairs of points from each sequence and look at how many are “concordant” or “discordant,” again similar to how Kendall’s Tau works. But then the process diverges: after ranking the data and considering the concordance and discordance among pairs, Hoeffding's D introduces a unique approach to quantify the dependency between the two sequences. It calculates a statistic based on the difference between the observed “joint distribution” of ranks and what would be expected if the two sequences were independent.

Let’s first take a step back and explain what we mean by “joint distribution” as compared to the individual or “marginal” distribution within each of our two sequences. In the context of Hoeffding's D, when we talk about the “joint distribution of ranks,” we're referring to how the ranks of the two sequences combine or relate to each other across all pairs of data points. Imagine plotting a graph where the x-axis represents the ranks from one sequence and the y-axis the ranks from the other. Each point on this graph then represents a pair of ranks— one from each sequence. The pattern these points form on the graph reflects their joint distribution: it shows us how the ranks in one sequence are associated with the ranks in the other sequence.

On the other hand, the “marginal distribution” pertains to the ranks within a single sequence, considered in isolation. If we were to look at just one axis of the aforementioned graph (either x or y), and ignore the other, the distribution of points along that axis would represent the marginal distribution for that sequence. It tells us about the spread or distribution of ranks within that sequence alone, without regard to how those ranks might be paired with ranks from the other sequence.

Understanding the distinction between joint and marginal distributions is crucial for grasping how Hoeffding's D works. The measure essentially evaluates whether the observed joint distribution of ranks deviates from what would be expected if the sequences were independent. Under independence, the joint distribution would simply be the product of the marginal distributions—meaning, the way ranks are paired across sequences would be random, with no discernible pattern. However, if there's a dependency between the sequences, the observed joint distribution will differ from this product of marginals, indicating that the ranks in one sequence are systematically related to the ranks in the other. Hoeffding's D quantifies this difference, providing a statistical measure of the dependency between the sequences.

In order to do this, Hoeffding's D considers all possible “quadruples” of rank pairs. That is, for any four distinct data points, it examines whether the ranking of one pair is consistent with the ranking of another pair within the sequences. This involves comparing each pair of data points against all other pairs, a process that is more comprehensive than the pairwise comparisons in Kendall's Tau, which looks only at concordant and discordant pairs. The essence of Hoeffding's D lies in its assessment of the joint ranking of the data points. It calculates the sum of certain terms derived from these comparisons, which reflect the degree of concordance and discordance across all pairs and quadruples. These terms account for the number of times a data point within one sequence is ranked both above and below another point when compared across both sequences, adjusted for ties.

The final computation of Hoeffding's D involves a formula that normalizes this sum, taking into account the total number of data points and the expected values under the assumption of independence. The result is a measure that ranges from -0.5 to 1, where the higher the number is, the more strongly dependent the two sequences are on each other. As you can probably imagine, computing Hoeffding’s D for two sequences of a certain length (say, 5,000 numbers in each of the two sequences) involves a huge number of individual comparisons and calculations— far more than are used to arrive at Kendall’s Tau, let alone Spearman’s Rho or Pearson’s correlation, since we are taking into account the entire sequence, not just individual pairs, but then also drilling down to the level of individual pairs.

For two sequences each containing 5,000 numbers, Hoeffding's D doesn't just compare each point with every other point once (which would already be substantial); it examines the relationships among all possible quadruples of points from the combined dataset. To put this into perspective, if you were to compare every pair of points in a single sequence of 5,000 numbers, you'd make about 12.5 million comparisons (since 5,000 choose 2 is approximately 12.5 million). But Hoeffding’s D requires comparing quadruples. The number of unique quadruples in a sequence of 5,000 is given by the combinatorial formula 'n choose 4', which for 5,000 is about 6.2 billion quadruples. And for each of these quadruples, Hoeffding’s D involves multiple comparisons and calculations to assess their concordance and discordance within the context of the entire data set.

This exponential increase in comparisons underscores why Hoeffding’s D is significantly more computationally demanding. It’s not merely a matter of scale; it’s the depth and breadth of analysis Hoeffding’s D performs to capture the complex dependencies between two sequences. This comprehensive approach allows Hoeffding's D to detect subtle and complex associations that simpler measures might miss, but it also means that its computation can be resource-intensive, particularly for large datasets. But I would argue that, in an age of cheap and fast computing, the time has come to start leveraging Hoeffding's D's many advantages. In addition to allowing us to find any kind of relationship between two sequences without making any assumptions about their distributions, and being robust to outliers, Hoeffding's D also has a few other benefits: it is symmetric, so that hoeffd(X, Y) is the same as hoeffd(Y, X), it is always bounded (the result will never be less than -0.5 or larger than 1.0), and when one of the sequences is a constant, the Hoeffding's D goes to zero. This is not true of some other powerful measures of association, such as *mutual information* (which we haven't discussed here). 

Now that we’ve given the basic overview, let’s get into the nitty gritty details of how this is all actually done. First I’ll give you the explanation in words with a single formula (this is the one place where trying to avoid using a formula would probably make things seem even more confusing and complicated than they already are!). Don’t worry if this is super confusing and makes little to no sense. It’s useful to see how the ideas are usually presented (trust me, the original 1948 paper is even more difficult to parse!). Then we will break it down piece by piece and try to give the intuition for each part. Finally we will look at an actual (but slow) implementation of Hoeffding’s D in Python, which is extensively commented.

**Step-by-Step Breakdown:**


1. **Ranking and Pairwise Comparisons**:
    - Initially, each data point in both sequences is replaced with its rank within the respective sequence, accounting for ties by assigning the average rank to tied values.
    - The comparison involves looking at all possible pairs within these ranked sequences to determine if they are concordant (both ranks increase or decrease together) or discordant (one rank increases while the other decreases).
    
2. **Quadruple Comparisons**:
    - Beyond pairwise comparisons, Hoeffding's D evaluates all possible quadruples of data points. For each quadruple, the measure assesses whether the ordering within one pair of ranks is consistent with the ordering within another pair. This step is crucial for capturing more complex dependencies beyond simple pairwise associations. We store these ranks in an array that we denote as **Q** which holds a value for each data point, reflecting its relation to other points in terms of rank concordance and discordance. For each data point, **Q** accumulates counts of how many pairs (quadruples when considered with another data point) show consistent (concordant) or inconsistent (discordant) ranking behavior. This step is essential for capturing complex dependencies that pairwise comparisons might miss.
    
3. **Summation**:
    - The core of Hoeffding's D calculation involves summing certain terms derived from the concordance and discordance assessments across all quadruples. This sum reflects the degree to which the observed joint distribution of ranks deviates from what would be expected if the sequences were independent. 
    
4. **Normalization**:
    - The final computation involves normalizing the sum obtained in the previous step. This normalization takes into account the total number of data points (**N**) and adjusts for the expected values under the assumption of independence. The point of this normalization is to scale the Hoeffding's D statistic, making it comparable across different sample sizes and distributions. 
    - The normalization formula is:
    ```math
     D = \frac{30 \times ((N-2)(N-3)D_1 + D_2 - 2(N-2)D_3)}{N(N-1)(N-2)(N-3)(N-4)}
    ```
    - Here, **D_1**, **D_2**, and **D_3** are intermediate sums that involve combinations of the ranks and their concordance/discordance assessments. Specifically:
        - **D_1** is related to the sum of products of differences in ranks for all quadruples, reflecting the overall concordance/discordance.
        - **D_2** adjusts for the individual variability within each sequence.
        - **D_3** accounts for the interaction between the sequences.

Since that all still looks pretty complicated, let’s get into what we mean by the quadruples and **Q** and the **D_1**, **D_2**, and **D_3** elements.

**What is Q and Its Purpose?**

- **Q** represents a weighted count for each data point, considering how many other points have ranks lower than its own in both dimensions (e.g., height and weight) and adjusting for ties.
- It's crucial for capturing the degree of concordance and discordance among the data points, going beyond simple pairwise comparisons.
- The computation of **Q** incorporates adjustments for ties, ensuring that tied ranks contribute appropriately to the overall measure.
- By considering all other points relative to a given point and incorporating adjustments for ties, **Q** values provide a nuanced view of each point's position within the joint distribution of ranks, which is foundational for assessing the dependency between the two sequences.

**Clarification of D_1, D_2, D_3**:

These intermediate sums play distinct roles in the calculation of Hoeffding's D:

  - **D_1** reflects the aggregated concordance/discordance among all quadruples of data points, adjusted for the expected amount under independence. It's a measure of how much the actual data deviates from what would be expected if there were no relationship between the sequences.
      - **Intuition for D_1**: Think of **D_1** as quantifying the extent of coordinated variation between the two sequences beyond what random chance would produce. It effectively captures the degree of mutual influence in the rank orderings, assessing whether the paired observations tend to move together in a consistent pattern more often than would be expected by coincidence.
      - **Very Simple Intuition for D_1**: Imagine you're comparing the synchronization of dance moves between two dancers across various performances. **D_1** represents the degree to which their moves are in sync (or out of sync) more than what you'd expect by mere chance if they were dancing independently. It captures the essence of their partnership by quantifying their coordinated (or uncoordinated) efforts.
  
  - **D_2** represents the product of rank variances within each sequence, offering a baseline measure of variability that's independent of any inter-sequence relationship. It helps isolate the effect of the sequences' internal variability from their mutual dependency.
      - **Intuition for D_2**: **D_2** evaluates the inherent variability or dispersion within each sequence on its own. It's akin to assessing the spread of data points in each sequence to understand how much each sequence varies independently. This helps differentiate between dependencies that arise from the inherent variability of the sequences themselves and those resulting from their interaction.
      - **Very Simple Intuition for D_2**: Consider **D_2** as assessing the range of dance styles each dancer showcases across performances, independent of their partner. It measures how varied or consistent each dancer is in their performances. By understanding this individual variability, we can better discern how much of their synchronization (captured by **D_1**) is due to their interaction rather than their individual tendencies.
  
  - **D_3** serves as an interaction term, blending the insights from **D_1** with the internal variability of the sequences captured by **D_2**. It fine-tunes the measure by accounting for how individual rank concordances contribute to the overall dependency, considering the internal rank structure of each sequence.
      - **Intuition for D_3**: **D_3** adjusts the measure by considering how the individual variability within each sequence (captured by **D_2**) influences the observed concordance/discordance (**D_1**). It's about understanding how the internal structure of each sequence impacts their joint behavior, refining the assessment of their relationship by accounting for the role of individual variability in their observed dependencies.
      - **Very Simple Intuition for D_3**: **D_3** is like examining how each dancer's individual style and consistency (**D_2**) influence their joint performance (**D_1**). If a dancer has a wide range of styles, how does this versatility affect their synchronization with their partner? **D_3** assesses the impact of each dancer's variability on their coordinated efforts, providing a nuanced view of their partnership.

         
**Final Hoeffding's D Calculation**:

- The final formula for Hoeffding's D combines **D_1**, **D_2**, and **D_3**, along with normalization factors, to produce a statistic that ranges from -0.5 to 1.
    - **Intuition for Final Calculation**: The final Hoeffding's D value is a comprehensive measure that integrates the observed concordance/discordance, the inherent variability within each sequence, and the interaction between these factors. The normalization ensures the measure is scaled appropriately for the sample size, making it a robust indicator of the strength and direction of the association between the sequences. It distills complex interdependencies and individual variabilities into a single metric that reflects the overall relationship, taking into account not just the presence of a relationship but also its nature and strength relative to what would be expected under independence.
    - **Very Simple Intuition for Final Calculation**: The final Hoeffding's D score is akin to a final score in a dance competition that evaluates the overall harmony between two dancers. This final score distills all aspects of their performances—individual styles, consistency, and mutual interaction—into a single, interpretable measure of their dance partnership. 

Although you probably have a much better idea of how Hoeffding’s D is computed now, things likely still seem abstract, and it’s not necessarily clear how you would actually implement the whole thing in a real programming language. So that’s what we’re going to do now in Python; to keep things short, we’re going to leverage the Numpy and Scipy libraries: Numpy for working with arrays effectively (including convenient indexing) and Scipy for its “rankdata” function, which computes ranks efficiently, handling ties that way we need using averaging. 

To make things really concrete, we are going to use just 10 data actual data points: they will be pairs of data of the form (`height_in_inches`, `weight_in_pounds`):


        X = np.array([55, 62, 68, 70, 72, 65, 67, 78, 78, 78])  # Heights in inches
        Y = np.array([125, 145, 160, 156, 190, 150, 165, 250, 250, 250])  # Weights in lbs

The result of running our program on that input is the following:


    Ranks of Heights (X): [1. 2. 5. 6. 7. 3. 4. 9. 9. 9.]
    Ranks of Weights (Y): [1. 2. 5. 4. 7. 3. 6. 9. 9. 9.]
    Q values: [1.  2.  4.  4.  7.  3.  4.  8.5 8.5 8.5]
    Hoeffding's D for data: 0.4107142857142857

Believe it or not, we can actually implement all of Hoeffding’s D in just ~15 lines of Python! However, we are going to add lots of comments and also display some intermediate values, because the focus is on making this as clear as possible. So we’ll end up with ~75 lines including whitespace— still not very much given how much the code is doing!

If you want to actually run the code, you can check out this Google Colab notebook here:


https://colab.research.google.com/drive/1MO_iheKH3syDgoYcWYzDuYKjaK8zyIi9?usp=sharing


Here is the complete code:

```python
    import numpy as np
    import scipy.stats
    from scipy.stats import rankdata
            
    def hoeffd_example():
        # New dataset of 10 data points representing pairs of (height_in_inches, weight_in_pounds), with the last 3 being ties:
        X = np.array([55, 62, 68, 70, 72, 65, 67, 78, 78, 78])  # Heights in inches
        Y = np.array([125, 145, 160, 156, 190, 150, 165, 250, 250, 250])  # Weights in pounds
        # The 'average' ranking method assigns the average of the ranks that would have been assigned to all the tied values.
        R = rankdata(X, method='average')  # Rank of heights
        S = rankdata(Y, method='average')  # Rank of weights
        # Print the ranks for visualization. Ties are evident in the last three entries of both rankings.
        print(f"Ranks of Heights (X): {R}")
        print(f"Ranks of Weights (Y): {S}")
        N = len(X)  # Total number of data points
        # Q is an array that will hold a special sum for each data point, which is crucial for Hoeffding's D computation.
        Q = np.zeros(N)
        # Loop through each data point to calculate its Q value.
        for i in range(N):
            # For each data point 'i', count how many points have both a lower height and weight rank (concordant pairs).
            Q[i] = 1 + sum(np.logical_and(R < R[i], S < S[i]))
            
            # Adjust Q[i] for ties: when both ranks are equal, it contributes partially (1/4) to the Q[i] value.
            # The "- 1" accounts for not including the point itself in its own comparison.
            Q[i] += (1/4) * (sum(np.logical_and(R == R[i], S == S[i])) - 1)
            
            # When only the height rank is tied but the weight rank is lower, it contributes half (1/2) to the Q[i] value.
            Q[i] += (1/2) * sum(np.logical_and(R == R[i], S < S[i]))
            
            # Similarly, when only the weight rank is tied but the height rank is lower, it also contributes half (1/2).
            Q[i] += (1/2) * sum(np.logical_and(R < R[i], S == S[i]))
        # Print the Q values for each data point, indicating the weighted count of points considered "lower" or "equal".
        print(f"Q values: {Q}")
        # Calculate intermediate sums required for Hoeffding's D formula:
        # D1: This sum leverages the Q values calculated earlier. Each Q value encapsulates information about how
        # a data point's ranks relate to others in both sequences, including concordance and adjustments for ties.
        # The term (Q - 1) * (Q - 2) for each data point quantifies the extent to which the ranks of this point
        # are concordant with others, adjusted for the expected concordance under independence.
        # Summing these terms across all data points (D1) aggregates this concordance information for the entire dataset.
        D1 = sum((Q - 1) * (Q - 2))
        # D2: This sum involves products of rank differences for each sequence, adjusted for ties. The term
        # (R - 1) * (R - 2) * (S - 1) * (S - 2) for each data point captures the interaction between the rank variances
        # within each sequence, providing a measure of how the joint rank distribution diverges from what would
        # be expected under independence due to the variability in ranks alone, without considering their pairing.
        # Summing these products across all data points (D2) gives a global assessment of this divergence.
        D2 = sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
        # D3: This sum represents an interaction term that combines the insights from Q values with rank differences.
        # The term (R - 2) * (S - 2) * (Q - 1) for each data point considers the rank variances alongside the Q value,
        # capturing how individual data points' rank concordance/discordance contributes to the overall dependency measure,
        # adjusted for the expected values under independence. Summing these terms (D3) integrates these individual
        # contributions into a comprehensive interaction term for the dataset.
        D3 = sum((R - 2) * (S - 2) * (Q - 1))
        # The final computation of Hoeffding's D integrates D1, D2, and D3, along with normalization factors
        # that account for the sample size (N). The normalization ensures that Hoeffding's D is scaled appropriately,
        # allowing for meaningful comparison across datasets of different sizes. The formula incorporates these sums
        # and normalization factors in a way that balances the contributions of concordance, discordance, and rank variances,
        # resulting in a statistic that robustly measures the degree of association between the two sequences.
        D = 30 * ((N - 2) * (N - 3) * D1 + D2 - 2 * (N - 2) * D3) / (N * (N - 1) * (N - 2) * (N - 3) * (N - 4))
        # Return the computed Hoeffding's D value.
        return D
    # Compute and display Hoeffding's D for the dataset
    hoeffd_d = hoeffd_example()
    print(f"Hoeffding's D for data: {hoeffd_d}")
```

Now, I mentioned that this implementation, while correct, is quite slow and inefficient. This might be surprising given that we are doing the “heavy lifting” using Numpy and Scipy, which are actually implemented in very efficient compiled C++. It turns out that, given the massive number of operations and comparisons involved, the function overhead is so great that the code really starts to bog down even on a fast machine once you get above 1,000 data points, and doing 5,000 data points would take an extremely long time. 

Luckily, I wrote a much more efficient version in Rust that is easy to use as a python library. You can use this simply by doing:

`pip install fast_vector_similarity` 

And you can see the code here:


https://github.com/Dicklesworthstone/fast_vector_similarity


And the discussion about the library on HN here:

https://news.ycombinator.com/item?id=37234887
