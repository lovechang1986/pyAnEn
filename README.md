# pyAnEn

## intro
AnEn算法的Python实现。想要使用C++和R语言的请访问[原生作者](https://github.com/Weiming-Hu/AnalogsEnsemble)
> 为什么说是原生作者呢？因为AnEn的发明者Luca教授也在其中。

> 个人感觉AnEn方法类似于KNN算法，总的思路大体一致，其应用于气象MOS或者后处理技术的几点创新如下：
> 	1. time windows，选取数值模式的前后两个预报时次(lead time)或更多；
> 	2. weight,在计算距离时就加入权重；
> 正因为如此，KNN所有的缺点在AnEn也同样不可避免的出现。
> 另外在KNN算法中得到最后投票结果时是可以根据距离进行权重的加权，但AnEn尚未发现有文章这么做。
>
> 可改进之处：
>
> 	1. 只在变量间加入权重，实际上个人感觉这是没太大必要的，因为查找相似是基于相似的天气形势，而天气形势或变量间往往是关联的，天气变量权重间的差异并没有想象的大。
>  	2. 天气形势间还有个更重要的维度就是时间，例如原方法中选取leadtime中前后时次做相似查找，但其实在不同的leadtime间加入权重并且当前leadtime占比最高，可能更有利于查找相似的时间变化。因为这种时间变化与天气变量对应可能更为重要。
>  	3. 以上纯属个人猜测未做对比。


## Usage

```
# AnEn class
from pyAnEn import AnEn

params = {
	'max_n_neighbours':50, 
	'weight_strategy':'total',
	'predict_name':'O3',
	'predictor_names':['O3', 't2', 'rh2', 'ws10', 'pblh'],
	'result_weight':'equal',
	'window_time':1,
}
anen_obj = AnEn(**params)

anen_obj.fit(trainx, trainy)
anen_obj.predict(testx)
```

## Contact

	- lyctze1986@gmail.com
	- add issues on Github

## Reference

- [2013 Luca](https://journals.ametsoc.org/doi/full/10.1175/MWR-D-12-00281.1)
