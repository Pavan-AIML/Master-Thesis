# In this folder we will implement the theory of model in to the coding,

![](Image/download.png)
![](Image/ad_4nxc9e-fxwecmh3e_zz4vgeiokxpdjuhexeca3dzysa2cf-choq0uchs39pdlqai6pd7lsdau3ptrjs_nofsuem1whkrxfogviyvlmdsjw05xzfpsljmzdyddua_t9hdnrrhup0mqpw0fualz4sdfachhgbak.avif)

### There are 4 different parts of this architecture as given in the research paper [Neural_Process](https://arxiv.org/pdf/1807.01622) and [Variational Auto Encoder](https://www.datacamp.com/de/tutorial/variational-autoencoders)
 


- [ ] Encoder : h(X<sub>c</sub>,Y<sub>c</sub>) -> r<sub>i</sub>
- [ ] Aggregator : a(r<sub>i</sub>) -> r
- [ ] Sampler : s(r) -> (&mu;(r), &sigma;(r)) ~ z
- [ ] Decoder : g(X<sub>T</sub>, z) -> &mu;, &sigma; of the target values with uncertainty.


From here decoder will produce the value of the 

## Data cleaning as we as deviding it to training and test data.

Once we have the cleaned and desired data we will devide it to the `contextual data`(X<sub>c</sub>, Y<sub>c</sub>) as well as `training data`(X<sub>T</sub>,Y<sub>T</sub>). 


## Encoder to generate the representation embeddings : 
### In this section we will look in to the working process of the encoder.


Contextual data X<sub>c</sub>, Y<sub>c</sub> values will be passed to the Encoder.

And the out-put embeddings will be stored in the form of representations r<sub>i</sub>

## <p align = "center">h(X<sub>c</sub>,Y<sub>c</sub>) -> r<sub>i</sub> </p>


## Aggregator :

This is the mean function for all the representations r<sub>i</sub>

## <p align = "center"> r =$\sum_{i=1}^{n} r_i$ </p>


## Latent Encoder for z : 

### In this section we will see that this encoder takes mean representation as an input and as an out-put it gives the mean and log variance that represents the distribution of z 

## <p align = "center">LE(r) = &mu;(z), log &sigma;(z)</p>


## Sampling z from here. 

### we consider $\epsilon \sim \mathcal{N}(0, 1)$ is normally distributed random variable. 

##  <p align = "center">$z \sim \mathcal{N}(\mu, \sigma^2)$ </p>

### Reparametrization technique :
## <p align = "center"> $z = \mu(z)+\epsilon*\sigma(z)$</p>

## Decoder :

### Once we have sampled z from here now this will be an output for the decoder model.

## <p align = "center"> $g(X_T, z) = \mu, log\sigma$</p>

## <p align ="center"> $var = \exp^{(2*log\sigma)}$</p>

## <p align = "center">$\mu = \mu$</p>


### As we can see above we have gained the mean and variance that provids us the target predicted values with some uncertainty. 

### Hence here we are not getting the deterministic values instead of that we are getting the distribution of the values. 



