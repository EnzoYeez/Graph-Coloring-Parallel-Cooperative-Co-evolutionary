# Charging Demand Estimation in the RPCS Model
In the RPCS model, a set of $\mathcal{N}$ candidate locations, roads $\mathcal{E}$, and scenarios $\mathcal{T}$ are given. 
The transportation network can be defined as:

$$
\mathcal{G}^{t} = \left(\mathcal{N}, \mathcal{E}^{t}\right), \quad t \in \mathcal{T}.
$$

An EV trip is characterized by an origin place $o \in \mathcal{N}$ and a destination place $d \in \mathcal{N}$, which forms an OD pair denoted as $q = (o,d)$.
The expected charging demand of all OD pairs on road $(i,j)$ in scenario $t$ is represented by the edge weight $w_{ij}^{t}$.

For an EV with battery capacity $E$ and energy consumption $e$ per kilometer, 

$$
s_{\min} = \frac{b}{E}
$$

represents the minimum reserve energy ratio. The initial state of charge, denoted by $s_{\mathrm{soc}}$, follows a truncated Beta distribution:

$$
s_{\mathrm{soc}} \sim \mathrm{TruncBeta}\left(\alpha,\beta; [s_{\min},1]\right).
$$

Its probability density function is given by:

$$
f(s_{\mathrm{soc}}) =
\begin{cases}
\dfrac{
s_{\mathrm{soc}}^{\alpha-1}
\left(1-s_{\mathrm{soc}}\right)^{\beta-1}
}{
B(\alpha,\beta)
\left[1-I_{s_{\min}}(\alpha,\beta)\right]
},
& s_{\mathrm{soc}} \in [s_{\min},1], \\[10pt]
0,
& \text{otherwise}.
\end{cases}
$$

where $\alpha$ and $\beta$ are the shape parameters of the Beta distribution, $B(\cdot)$ is the Beta function, and $I_{s_{\min}}(\cdot)$ is the incomplete Beta function.

For an OD pair $q=(o,d)$ traversing through the locations $\{0,1,\dots,m\}$ in scenario $t$, the road length from location $k-1$ to location $k$ is denoted by $L_k$.

If the EV departs with an initial SoC $s_{\mathrm{soc}}$, a charging event is triggered between location $m-1$ and location $m$ for OD pair $q$ in scenario $t$ if

$$
E s_{\mathrm{soc}}
-
e \sum_{k=0}^{m} L_k
\le b
\quad \text{and} \quad
E s_{\mathrm{soc}}
-
e \sum_{k=0}^{m-1} L_k
> b.
$$

This indicates that the vehicle can reach location $m-1$ but cannot continue beyond location $m$ without charging.

This condition corresponds to the following SoC interval:

$$
s_{\mathrm{soc}}
\in
\left(
\frac{
b + e \sum_{k=0}^{m-1} L_k
}{E},
\frac{
b + e \sum_{k=0}^{m} L_k
}{E}
\right].
$$

Then, the probability that the first charging event for OD pair $q$ occurs between location $m-1$ and location $m$ during scenario $t$ is:

$$
p_{(m-1,m),q}^{t}
=
F\left(
\frac{
b + e \sum_{k=0}^{m} L_k
}{E}
\right)
-
F\left(
\frac{
b + e \sum_{k=0}^{m-1} L_k
}{E}
\right),
$$

where $F(\cdot)$ is the cumulative distribution function of $s_{\mathrm{soc}}$.

Similarly, the charging probabilities for all roads along the path, i.e., $(0,1), (1,2), \dots, (m-2,m-1)$, can be derived in the same manner.

The charging demand of edge $(i,j)$ is obtained by summing over all OD pairs $Q_{ij}^{t}$ in scenario $t \in \mathcal{T}$ as:

$$
w_{ij}^{t}
=
\sum_{q \in Q_{ij}^{t}}
p_{ij,q}^{t},
$$

where $p_{ij,q}^{t}$ is the probability that the charging event of OD pair $q$ occurs on the segment between location $i$ and location $j$ during scenario $t$.


### Parameters Used in the Charging Probability Model



Except for OD pairs and their corresponding path information, we provide all key parameters required by the proposed charging probability model. These parameters are summarized as follows. 

| Mathematical Symbol | Value | Description |
|:---:|:---:|:---:|
| $\alpha$ | $3.28$ | Shape parameter of the truncated Beta distribution for the initial SoC. |
| $\beta$ | $3.27$ | Shape parameter of the truncated Beta distribution for the initial SoC. |
| $s_{\min}$ | $0.10$ | Lower bound of the truncated SoC distribution. |
| $E$ | $78.1$ kWh | Battery capacity of the EV. |
| $e$ | $25.0 / (100.0 \times 1.609344)$ kWh/km | Energy consumption per kilometer. |
| $b/E$ | $0.10$ | Minimum reserve energy ratio. |
| $b$ | $7.81$ kWh | Minimum reserve energy required by the EV. |

 Therefore, once the OD pairs, paths, and edge distances are provided, the code contains sufficient parameters to compute the truncated-Beta-based first charging probability and aggregate the edge weights.