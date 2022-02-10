import numpy as np

L_max = 1e10
print_iter = 50
level_contour = 70

def soft_threshold_nonneg_healpix(x_d, eta,  crit=0.1, crit_flag = False):
    vec = np.zeros(np.shape(x_d))
    mask=x_d>eta
    vec[mask]=x_d[mask] - eta
    return vec

## Calculation of TSV
def TSV_healpix(weight, image):
    return np.dot( image, np.dot(weight,  image))

## Calculation of d_TSV
def d_TSV_healpix(weight, image):
    return 2 * np.dot(weight,  image)


def F_TSV_healpix(data, W,  x_d, lambda_tsv, weight, sigma):
    data_dif = data -  np.dot(W, x_d)
    if lambda_tsv == 0:
        return (np.dot(data_dif, (1/sigma**2) *data_dif)/2)
    return (np.dot(data_dif, (1/sigma**2) *data_dif)/2)  + TSV_healpix(weight, x_d) *  lambda_tsv

def F_obs_healpix(data, W,  x_d, sigma):
    data_dif = data -  np.dot(W, x_d)
    return (np.dot(data_dif, (1/sigma**2) *data_dif)/2)

# Derivative of ||y-Ax||^2 + TSV (F_TSV)
##  np.dot(A.T, data_dif) is n_image vecgtor, d_TSV(x_d) is the n_image vecgtor or matrix

def dF_dx_healpix(data, W, x_d, lambda_tsv, weight, sigma):
    data_dif = -(data -  np.dot(W, x_d))
    if lambda_tsv == 0:
        return np.dot(W.T, (1/sigma**2) *data_dif)

    return np.dot(W.T, (1/sigma**2) *data_dif) +lambda_tsv*  d_TSV_healpix(weight, x_d)

## Calculation of Q(x, y) (or Q(P_L(y), y)) except for g(P_L(y))
## x_d2 = PL(y) (xvec1)
## x_d = y (xvec2)

def calc_Q_part_healpix(data, W,  x_d2, x_d, df_dx, L, lambda_tsv, weight, sigma):
    Q_core = F_TSV_healpix(data, W, x_d, lambda_tsv, weight, sigma)
    Q_core += np.dot((x_d2 - x_d), df_dx) + 0.5 * L * np.dot(x_d2 - x_d, x_d2 - x_d)
    return Q_core


def mfista_func_healpix(I_init, d, A_ten, weight,sigma, lambda_l1= 1e2, lambda_tsv= 1e-8, L_init= 1, eta=1.02, maxiter= 200, max_iter2=100, 
                    miniter = 2000, TD = 30, eps = 1e-5, log_flag = False, do_nothing_flag = False, prox_map = soft_threshold_nonneg_healpix, prox_crit=0.1, prox_crit_flag=False):

    if do_nothing_flag == True:
        return I_init

    ## Initialization
    mu, mu_new = 1, 1
    y = I_init
    x_prev = I_init
    cost_arr = []
    L = L_init
    x_best = I_init

    ## The initial cost function
    cost_first = F_TSV_healpix(d, A_ten, I_init, lambda_tsv, weight, sigma)
    cost_first += lambda_l1 * np.sum(np.abs(I_init))
    cost_temp, cost_prev = cost_first, cost_first
    cost_min = cost_temp  * 1e10

    
    ## Main Loop until iter_now < maxiter
    ## PL_(y) & y are updated in each iteration
    for iter_now in range(maxiter):
        cost_arr.append(cost_temp)
        
        ##df_dx(y)
        df_dx_now = dF_dx_healpix(d, A_ten, y, lambda_tsv, weight, sigma) 
        
        ## Loop to estimate Lifshitz constant (L)
        ## L is the upper limit of df_dx_now
        for iter_now2 in range(max_iter2):
            
            y_now = prox_map(y - (1/L) * df_dx_now, lambda_l1/L,  prox_crit, prox_crit_flag)
            Q_now = calc_Q_part_healpix(d, A_ten, y_now, y, df_dx_now, L,  lambda_tsv, weight, sigma)
            F_now = F_TSV_healpix(d, A_ten, y_now,lambda_tsv, weight, sigma)
            
            ## If y_now gives better value, break the loop
            if F_now <Q_now:
                break

            L = L*eta

        #L = L/eta

        
        if L > L_max:
            logger.info("L:%e,  Too much large L!!!" % L)
            break
        
        mu_new = (1+np.sqrt(1+4*mu*mu))/2
        F_now += lambda_l1 * np.sum(np.abs(y_now))

        ## Updating y & x_k
        if F_now < cost_prev:
            cost_temp = F_now
            tmpa = (1-mu)/mu_new
            x_k = prox_map(y - (1/L) * df_dx_now, lambda_l1/L,  prox_crit, prox_crit_flag)
            y = x_k + ((mu-1)/mu_new) * (x_k - x_prev) 
            x_prev = x_k

            if F_now < cost_min:
                cost_min = F_now
                x_best = y_now
 
        else:
            cost_temp = F_now
            tmpa = 1-(mu/mu_new)
            tmpa2 =(mu/mu_new)
            x_k = x_prev
            z = prox_map(y - (1/L) * df_dx_now, lambda_l1/L, prox_crit, prox_crit_flag)
            y = tmpa2 * z + tmpa * x_prev 
            x_prev = x_k
        #logger.debug("iter_now %d, L: %f, F_sum:%f, l1_term:%f" % (iter_now, L, F_now, lambda_l1 * np.sum(np.abs(y_now))))
        if iter_now % print_iter  == 0:
            if lambda_tsv == 0:
                tsv_term = 0
            else:
                tsv_term = lambda_tsv * TSV_healpix(weight,y)
            log_out_now = "l1:%.2f, ltsv:%.2f, Current iteration: %d/%d,  L: %f, cost: %f, cost_chiquare:%f, cost_l1:%f, cost_ltsv:%f" % (np.log10(lambda_l1), np.log10(lambda_tsv), iter_now, maxiter, L, cost_temp, F_obs_healpix(d, A_ten, y, sigma),lambda_l1 * np.sum(np.abs(y)),tsv_term)
            logger.info(log_out_now)
        if(iter_now>miniter) and cost_arr[iter_now-TD]-cost_arr[iter_now]<cost_arr[iter_now]*eps:
            break

        mu = mu_new
    
    #print('L=',L)

    if lambda_tsv == 0:
        tsv_term = 0
    else:
        tsv_term = lambda_tsv * TSV_healpix(weight,y)
    log_out_now = "l1:%.2f, ltsv:%.2f, Current iteration: %d/%d,  L: %f, cost: %f, cost_chiquare:%f, cost_l1:%f, cost_ltsv:%f" % (lambda_l1, lambda_tsv, iter_now, maxiter, L, cost_temp, F_obs_healpix(d, A_ten, y, sigma),lambda_l1 * np.sum(np.abs(y)),tsv_term)
    if log_flag:
        logger.debug(log_out_now)
    return x_best

## Cauculation of matrix to express neighboring pixels 
## This is used for calculation of TSV & d_TSV
def calc_neighbor_weightmatrix(hp, nside):
    nside_now = nside
    Npix = 12 * nside **2
    Neighbor_matrix = np.zeros((Npix,Npix))
    Weight_tsv_matrix = np.zeros((Npix,Npix))
    for i in range(Npix):
        neighbor = hp.get_all_neighbours(nside_now, i)
        for j in range(8):
            neighbor_ind = neighbor[j]
            if neighbor_ind == -1:
                continue
            Neighbor_matrix[i][neighbor_ind] = 1
            Weight_tsv_matrix[i][i] += 0.5
            Weight_tsv_matrix[i][neighbor_ind] -= 0.5
            Weight_tsv_matrix[neighbor_ind][i] -= 0.5
            Weight_tsv_matrix[neighbor_ind][neighbor_ind] += 0.5
    return Weight_tsv_matrix,Neighbor_matrix
