# The Vicsek model

This model looks into how flocks of birds or swarms of fish move together so coherantly in large groups. Research indicates that the true cause is local interactions with individuals within the flock.

## Modelling

Each bird will be represented by a small point particle, that moves with a constant velocity $v_0$ along it's polar angle $\vec{\bold{n}}(\theta) = (cos(\theta),sin(\theta))$. At each discrete time step $k$ the bird checks its neighbours within a radius $R$ around themselves and then re-orients it's self along their mean direction. In a 2D plane the polar equations of motion are given by:

$$
\vec{\bold{r}}_i(k+1) = \vec{\bold{r}}_i(k) + \vec{\bold{n}}v_0
\\
\vec{\bold{\theta}}_i(k+1) = angle[\sum_{j \in neighbours}\vec{\bold{n}}_j] + \mu_i
$$

$\mu_i$ is chosen from a normal distribution with mean $0$ and standard deviation of $\sigma$ for each bird, i.

The Vicsek order paramater is calculated as follows:

$$
n = \frac{1}{N}|\sum^{N}_{i = 1}\vec{\bold{n}}_i|
$$

i.e the mean direction.

## How to run

In the command line run:

```sh
python run_animation.py
```

This will save an MP4 file of the model outputs.