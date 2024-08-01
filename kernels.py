import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.stats import norm
import jax
import math

def get_kernels(sign,distance,kernel,blurriness,dist_shape=0.5,dist_shift=0.):
    GAMMA_THRESHOLD = 15.
    NUM_STEPS_GAMMA = 32

    # finite support
        # exact
    if kernel == 'hard':
        return (sign>0)
        # rasterise = jnp.where(sign>0,sign,0)

        # continuous
    elif kernel == 'uniform':
        # if(sign * distance / blurriness < -1):
        #     return 0
        # elif(sign * distance / blurriness < 1):
        #     return (sign * distance) * 0.5 / blurriness + 0.5
        # else:
        #     return 1.
        result = jnp.where(sign*distance/blurriness<1,(sign * distance) * 0.5 / blurriness + 0.5,jnp.ones_like(sign))
        result = jnp.where(sign*distance/blurriness<-1,0.,result)
        return result

    elif kernel == 'cubic_hermite':
        # if (sign * distance / blurriness < -1):
        #     return 0.
        # elif (sign * distance / blurriness < 1): 
        #     y = (sign * distance) * 0.5 / blurriness + 0.5
        #     return 3 * y * y - 2 * y * y * y
        # else:
        #     return 1.
        y = (sign * distance) * 0.5 / blurriness + 0.5
        result = jnp.where(sign*distance/blurriness<1,3 * y * y - 2 * y * y * y,jnp.ones_like(sign))
        result = jnp.where(sign*distance/blurriness<-1,0.,result)
        return result

    elif kernel == 'wigner_semicircle':
        # if (sign * distance / blurriness < -1):
        #     return 0.
        # elif (sign * distance / blurriness < 1):
        #     return 0.5 + (sign * distance * jnp.sqrt(blurriness * blurriness - distance * distance)) / (jnp.pi * blurriness * blurriness) + jnp.asin(sign * distance / blurriness) / jnp.pi
        # else:
        #     return 1.
        result = jnp.where(sign*distance/blurriness<1,0.5 + (sign * distance * jnp.sqrt(blurriness * blurriness - distance * distance)) / (jnp.pi * blurriness * blurriness) + jnp.arcsin(sign * distance / blurriness) / jnp.pi,jnp.ones_like(sign))
        result = jnp.where(sign*distance/blurriness<-1,0.,result)
        return result

    # infinite support 
        # symmetrical
            # exponential conv
    elif kernel == 'gaussian':
        return norm.cdf(sign * distance / blurriness)

    elif kernel == 'logistic':
        # rasterise = 1. / (1. + jnp.exp(- sign * distance / blurriness))
        return jax.nn.sigmoid(sign * distance / blurriness)

    elif kernel == 'square_logistic':
        # rasterise = 1. / (1. + jnp.exp(- sign * distance / blurriness))
        return jax.nn.sigmoid(sign * (distance)**2 / blurriness)

    elif kernel == 'laplace':
        # if (sign < 0):
        #     return 0.5 * jnp.exp(- distance / blurriness)
        # else:
        #     return 1. - 0.5 * jnp.exp(- distance / blurriness)
        return jnp.where(sign<0,0.5 * jnp.exp(- distance / blurriness),1. - 0.5 * jnp.exp(- distance / blurriness))
        
    elif kernel == 'gudermannian':
        return jnp.arctan(jnp.tanh(sign * distance / blurriness / 2.)) * 2. / jnp.pi + 0.5

            # linear conv
    elif kernel == 'reciprocal':
        return sign * distance / blurriness / (1 + distance / blurriness) / 2. + 0.5

    elif kernel == 'cauchy':
        return jnp.arctan(sign * distance / blurriness) / jnp.pi + 0.5

        # asymmetrical
            # two-sided
    elif kernel == 'gumbel_max':
        return jnp.exp(-jnp.exp(- sign * distance / blurriness))

    elif kernel == 'gumbel_min':
        return 1. - jnp.exp(-jnp.exp(- sign * distance / blurriness))
            # one-sided

    elif kernel == 'square_exponential':
        # return jnp.exp(-(distance)**2/blurriness)
        return get_exponential(sign*distance**2,blurriness)

    elif kernel == 'exponential':
        return jnp.where((sign * distance + dist_shift * blurriness < 0.),0.,1. - jnp.exp(- (sign * distance + dist_shift * blurriness) / blurriness))
        # return get_exponential(sign*distance,blurriness)

    elif kernel == 'exponential_neg':
        return jnp.where((sign * distance - dist_shift * blurriness > 0.),1.,jnp.exp((sign * distance - dist_shift * blurriness) / blurriness))
    
    elif kernel == 'gamma':
        if (dist_shape < 0.):
            print("Error in kernel gamma; invalid param p (dist_shape): %g\n", dist_shape);
            return jnp.nan
        return jsp.stats.gamma.cdf(sign * distance, a=dist_shape, scale=blurriness)
        # return get_gamma_neg(sign*distance,blurriness)
        # xs = sign * distance + dist_shift * blurriness
        # # kummers = 1. / jax.scipy.stats.gamma.pdf(dist_shape + 1.)
        # kummers = 1. / math.gamma(dist_shape + 1.)
        # factor = kummers
        # for i in range(1,NUM_STEPS_GAMMA):
        #     factor *= xs / blurriness / (dist_shape + i)
        #     kummers += factor
        
        # y = pow(xs / blurriness, dist_shape) * jnp.exp(- xs / blurriness) * kummers
        # result = jnp.where(xs / blurriness > GAMMA_THRESHOLD,1.,y)
        # return jnp.where(xs<=0.,0.,result)
    
    elif kernel == 'gamma_neg':
        if (dist_shape < 0.):
            print("Error in kernel gamma; invalid param p (dist_shape): %g\n", dist_shape);
            return jnp.nan
        return 1.-jsp.stats.gamma.cdf(-sign * distance, a=0.5, scale=blurriness)
        # xs = - (sign * distance - dist_shift * blurriness)
        # # kummers = 1. / jax.scipy.stats.gamma.pdf(dist_shape + 1.)
        # kummers = 1. / math.gamma(dist_shape + 1.)
        # factor = kummers
        # for i in range(1,NUM_STEPS_GAMMA):
        #     factor *= xs / blurriness / (dist_shape + i)
        #     kummers += factor
        
        # y = pow(xs / blurriness, dist_shape) * jnp.exp(- xs / blurriness) * kummers
        # result = jnp.where(xs / blurriness > GAMMA_THRESHOLD,0.,1.-y)
        # return jnp.where(xs<=0.,1.,result)

    elif kernel == 'levy':
        xs = sign * distance + dist_shift * blurriness
        y = jax.lax.erfc(jnp.sqrt(blurriness / 2. / xs))
        return jnp.where(xs<=1e-6,0.,y)
    elif kernel == 'levy_neg':
        xs = - (sign * distance - dist_shift * blurriness)
        y = jax.lax.erfc(jnp.sqrt(blurriness / 2. / xs))
        return jnp.where(-xs>=-1e-6,1.,1.-y)
    else:
        assert False, "no kernel assigned"

    return rasterise

@jax.custom_jvp
def get_exponential(sign_distance,blurriness):
    return jnp.where((sign_distance < 0.),0.,1. - jnp.exp(- (sign_distance) / blurriness))

@get_exponential.defjvp
def get_exponential_jvp(primals, tangents):
    sign_distance,blurriness = primals
    sign_distance_dot,blurriness_dot = tangents
    primal_out = get_exponential(sign_distance,blurriness)
    tangent_out = jnp.where((sign_distance < 0.),0.,sign_distance_dot * sign_distance / blurriness * jnp.exp(- sign_distance/ blurriness) + blurriness_dot * sign_distance * jnp.exp(- sign_distance/ blurriness))
    return primal_out, tangent_out

@jax.custom_jvp
def get_gamma_neg(sign_distance,blurriness):
    xs = - (sign_distance - 0 * blurriness)
    kummers = 1. / math.gamma(0.5 + 1.)
    factor = kummers
    for i in range(1,32):
        factor *= xs / blurriness / (0.5 + i)
        kummers += factor
    
    y = pow(xs / blurriness, 0.5) * jnp.exp(- xs / blurriness) * kummers
    result = jnp.where(xs / blurriness > 15.,0.,1.-y)
    return jnp.where(xs<=0.,1.,result)

@get_gamma_neg.defjvp
def get_gamma_neg_jvp(primals, tangents):
    sign_distance,blurriness = primals
    sign_distance_dot,blurriness_dot = tangents
    primal_out = get_gamma_neg(sign_distance,blurriness)
    
    tangent_out = jnp.where((sign_distance >= 0.),0.,sign_distance_dot * pow(1. /  blurriness, 0.5) / math.gamma(0.5)
                * pow(-sign_distance, 0.5 - 1.) * jnp.exp(sign_distance / blurriness) + blurriness_dot * sign_distance * pow(1. /  blurriness,  0.5 + 1.) / math.gamma(0.5)
                * pow(-sign_distance,  0.5 - 1.) * jnp.exp(sign_distance /  blurriness))
    return primal_out, tangent_out