# In neural process we will try to understand the loss functions and theory behind it. 

# Goal : To maximize the log liklihood.

## The objective is to maximize the liklihood to get Y<sub>c</sub>. 

## Objective : Maximize the `marginal liklihood`

## <p align = "center"> $max  \log p(y_t /x_t, y_c,x_c)$</p>

## The above integral can be written in the form of latent variable z as this is also involve in this.

## <p align = "center"> $\log p(y_t/x_t, x_c, y_c)= \log \int p(y_t/x_t,z)p(z/x_c, y_c) dz$ </p>

## Calculating above integral is intractable as z is in the higher dimension. 

## Hence from here we will try to solve this in some other way. Instead of maximizing this we will maximize it's lower bound. 


## <p align = "center" > $\log p(y_t/x_t,x_c, y_c) \ge E_{q(z/x_T, y_T, x_C, y_C)}[\log p(y_T/x_T, z)]-D_{KL}(q(z / x_C, y_C)||p(z))$</p>

## <p align = "center" > $\log p(y_t/x_t,x_c, y_c) \ge ELBO$</p>


## <p> Here the first term $E_{q(z)}[\log p(y_T/x_T, x_C, y_C)]$ measures how well the model fits the target data. This is called the reconstruction loss $L_{recon.}$</p>

## The second term $D_{KL}(q||p)$ is measures how close the posterier to the prior distribution over z.

## Total objective function :

## <p align = "center" > max $L_{NP} =L_{recons} - \beta *L_{KL}$</p>

## In pytorch we minimize the objective function or loss function. 

## min<p align = "center" >$- L_{NP} = - L_{recons} + \beta *L_{KL}$</p>

## Hence the loss :

## $L_{NP} =NLLH (Negative log liklihood) (- term will absorbe in this) + \beta * KL_{loss}$

## We will se here the loss function, reconstruction Negative log Liklihood loss.


## <p align = "center"> $NLL = 0.5 * \sum_{i=1}^{n} (y_i -u_{yi})^2 / \sigma_i^2 + \sum_{i=1}^{n} log(2 * \pi * \sigma_{y_i}^2)$</p>

## [KL_Divergence](https://www.datacamp.com/tutorial/kl-divergence?utm_cid=19589720821&utm_aid=157156374951&utm_campaign=230119_1-ps-other~dsa~tofu_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9061166-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~emea-en~dsa~tofu~tutorial~machine-learning&gad_source=1&gad_campaignid=19589720821&gbraid=0AAAAADQ9WsHcPUoVAsxPWU3nzTSP3Q_RE&gclid=CjwKCAiAraXJBhBJEiwAjz7MZeYTpHWrqM2nAdfOMdlOfiajMt4r-3BD5X5bReBEZfUOLVFF3X-ZExoCtV8QAvD_BwE)

## <p align = "center"> $p(z) = \mathcal{N(\mu, \sigma^2 I)}$</p>
## <p align = "center">$D_{KL}(p||q) = \sum_{i=1}^{n}$</p>

