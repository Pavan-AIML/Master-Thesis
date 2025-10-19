# In neural process we will try to understand the loss functions and theory behind it. 

# Goal : To maximize the log liklihood.

## The objective is to maximize the liklihood to get Y<sub>c</sub>. 

## Objective : Maximize the `marginal liklihood`

## <p align = "center"> $max  \log p(y_t /x_t, y_c,x_c)$</p>

## The above integral can be written in the form of latent variable z as this is also involve in this.

## <p align = "center"> $\log p(y_t/x_t, x_c, y_c)= \log \int p(y_t/x_t,z)p(z/x_c, y_c) dz$ </p>

## Calculating above integral is intractable as z is in the higher dimension. 

## Hence from here we will try to solve this in some other way. Instead of maximizing this we will maximize it's lower bound. 


## <p align = "center" > $\log p(y_t/x_t,x_c, y_c) \ge E_{q(z/x_T, y_T, x_C, y_C)}[\log p(y_T/x_T, z)]-D_{KL}(q(z/x_T, y_T, x_C, y_C)||p(z/x_C,y_C))$</p>

## <p align = "center" > $\log p(y_t/x_t,x_c, y_c) \ge ELBO$</p>


## <p> Here the first term $E_{q(z)}[\log p(y_T/x_T, x_C, y_C)]$ measures how well the model fits the target data. This is called the reconstruction loss $L_{recon.}$</p>

## The second term $D_{KL}(q||p)$ is measures how close the posterier to the prior distribution over z.

## Total objective function :

## <p align = "center" > max $L_{NP} =L_{recons} - \beta *L_{KL}$</p>

## In pytorch we minimize the objective function or loss function. 

## min<p align = "center" >$- L_{NP} = - L_{recons} + \beta *L_{KL}$</p>

## Hence the loss :

## $L_{NP} =NLLH (Negative log liklihood) (- term will absorbe in this) + \beta * KL_{loss}$

