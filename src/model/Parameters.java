package model;

/**
 * Created by tndoan on 4/25/17.
 */
public class Parameters {

    /**
     * regularizer parameter for user vector
     */
    private double lambda_u;

    /**
     * regularizer parameter for venue vector
     */
    private double lambda_v;

    /**
     * regularizer parameter for friendship
     */
    private double lambda_f;

    /**
     *
     * @param lambda_u  regularizer parameter for user vector
     * @param lambda_v  regularizer parameter for venue vector
     * @param lambda_f  regularizer parameter for friendship
     */
    public Parameters(double lambda_u, double lambda_v, double lambda_f) {
        this.lambda_f = lambda_f;
        this.lambda_u = lambda_u;
        this.lambda_v = lambda_v;
    }

    /**
     *
     * @return  regularizer of user
     */
    public double getLambda_u() {
        return lambda_u;
    }

    /**
     *
     * @return  regularizer of venue
     */
    public double getLambda_v() {
        return lambda_v;
    }

    /**
     *
     * @return  regularizer of friendship
     */
    public double getLambda_f() {
        return lambda_f;
    }
}
