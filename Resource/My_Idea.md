## Relatedness
* 问句替换中，单纯对entity找替换对象和单纯对relation找替换对象都是不正确的，因为
比如牛顿的相似实体，可能会想到爱因斯坦等等，但是当我们问牛顿的导师是谁的时候，
可能牛顿的同班同学才是最合适的
* 不同的领域有不通的相关，所以知识图谱中的相关应该是要加上relation的影响。
* 相似性这个东西无法解决，因为我们要求的是不同领域的相似性，比如
    * Newton doctorialAdvisor ？替换的实体最好是Newton的同学
    * ? doctorialAdvisor Newton 替换最好的实体可能就是牛顿的同事
    * 传统方法无法真正解决这个问题，没有办法在每个关系和每个位置都给一个真正相似的实体
    * 用Embedding，来解决这个问题，当我们要Newton doctorialAdvisor ？
    * 得到Newton + doctorialAdvisor的embedding，
    * 其余的和Newton相关的实体都加上doctorialAdvisor的embedding，然后选出相关的实体
    * 已有的方法