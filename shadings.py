import numpy as np
import jax.numpy as jnp
from util_parallel import normalize
import jax

def flat_shading(mesh,normal,depth,depth_inv,material_Ns,material_K,light_position,light_intensity,cam_pos):
    view_points = mesh[:,:,:3].mean(1) # [n,3]
    normal = normal[:,:,:3].mean(1) # [n,3]
    n = normalize(normal,axis=-1)
    v = normalize(cam_pos - view_points,axis=-1) # [n,3]
    l = normalize(light_position - view_points,axis=-1) # [n,3]
    h = normalize(l+v,axis=-1)
    r2 = jnp.linalg.norm(light_position - view_points,axis=-1)**2
    Ns = material_Ns
    Ka = material_K[0]
    Kd = material_K[1]
    Ks = material_K[2]
    ambient = Ka[None,:]*light_intensity
    diffuse = Kd[None,:]*((light_intensity/r2)*jnp.fmax(0.,jnp.einsum('il,il->i',n,l)))[:,None]
    specular = Ks[None,:]*(((light_intensity/r2)*pow(jnp.fmax(0.,jnp.einsum('il,il->i',n,h)),Ns)))[:,None]
    return (ambient + diffuse + specular)[:,None,None,:] #[n,1,1,3]

def gourard_shading(mesh,normal,depth,depth_inv,material_Ns,material_K,light_position,light_intensity,cam_pos):
    view_points = mesh[:,:,:3] # [n,3,3]
    normal = normal[:,:,:3] # [n,3,3]
    n = normalize(normal,axis=-1)
    v = normalize(cam_pos - view_points,axis=-1) # [n,3,3]
    l = normalize(light_position - view_points,axis=-1) # [n,3,3]
    h = normalize(l+v,axis=-1)
    r2 = jnp.linalg.norm(light_position - view_points,axis=-1)**2
    Ns = material_Ns
    Ka = material_K[0]
    Kd = material_K[1]
    Ks = material_K[2]
    ambient = Ka[None,None,:]*light_intensity
    diffuse = Kd[None,None,:]*((light_intensity/r2)*jnp.fmax(0.,jnp.einsum('ijk,ijk->ij',n,l)))[:,:,None]
    specular = Ks[None,None,:]*(((light_intensity/r2)*pow(jnp.fmax(0.,jnp.einsum('ijk,ijk->ij',n,h)),Ns)))[:,:,None]
    color = ambient + diffuse + specular #[n,3,3]
    return jax.lax.stop_gradient(depth)*(color[:,:,None,None,:]*jax.lax.stop_gradient(depth_inv)).sum(axis=1) #[n,h,w,3]


def phong_shading(mesh,normal,depth,depth_inv,material_Ns,material_K,light_position,light_intensity,cam_pos):
    normal = jax.lax.stop_gradient(depth)*(normal[:,:,None,None,:]*jax.lax.stop_gradient(depth_inv)).sum(axis=1) #[n,h,w,3]
    view_points_xy = jax.lax.stop_gradient(depth)*(mesh[:,:,:2][:,:,None,None,:]*jax.lax.stop_gradient(depth_inv)).sum(axis=1) 
    view_points = jnp.concatenate([view_points_xy,jax.lax.stop_gradient(depth)],axis=-1) #[n,h,w,3]
    n = normalize(normal,axis=-1)
    v = normalize(cam_pos - view_points,axis=-1) # [n,h,w,3]
    l = normalize(light_position - view_points,axis=-1) # [n,h,w,3]
    h = normalize(l+v,axis=-1)
    r2 = jnp.linalg.norm(light_position - view_points,axis=-1)**2 #[n,h,w]
    Ns = material_Ns
    Ka = material_K[0]
    Kd = material_K[1]
    Ks = material_K[2]
    ambient = Ka[None,None,None,:]*light_intensity
    diffuse = Kd[None,None,None,:]*((light_intensity/r2)*jnp.fmax(0.,jnp.einsum('ijkl,ijkl->ijk',n,l)))[:,:,:,None]
    specular = Ks[None,None,None,:]*(((light_intensity/r2)*pow(jnp.fmax(0.,jnp.einsum('ijkl,ijkl->ijk',n,h)),Ns)))[:,:,:,None]
    return ambient + diffuse + specular