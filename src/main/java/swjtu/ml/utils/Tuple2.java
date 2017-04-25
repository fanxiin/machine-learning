package swjtu.ml.utils;

/**
 * Created by xin on 2017/4/17.
 */
public class Tuple2<E, T> {
    private E e;
    private T t;

    public Tuple2(E e, T t){
        this.e = e;
        this.t = t;
    }

    public E _1(){
        return e;
    }

    public T _2(){
        return t;
    }
}