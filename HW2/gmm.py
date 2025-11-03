import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm

SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = True


class GMM(object):
    def __init__(self, X, K, max_iters=100):
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]
        self.D = self.points.shape[1]
        self.K = K
        self.num_iters = 1

    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error.
        """
        centered_logit = logit - np.max(logit, axis=1, keepdims=True)
        exp_logit = np.exp(centered_logit)
        prob = exp_logit / np.sum(exp_logit, axis=1, keepdims=True)
        return prob


    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        max_logit = np.max(logit, axis=1, keepdims=True)
        centered_logit = logit - max_logit
        s = max_logit + np.log(np.sum(np.exp(centered_logit), axis=1, keepdims=True))
        return s


    def normalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        D = points.shape[1]
        diff = points - mu_i
        variances = np.diagonal(sigma_i)
        
        pdf = np.prod(1.0 / np.sqrt(2 * np.pi * variances) * np.exp(-0.5 * (diff**2) / variances), axis=1)
        return pdf


    def multinormalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            1. np.linalg.det() and np.linalg.inv() should be handy.
            2. Note the value in self.D may be outdated and not correspond to the current dataset.
            3. You may wanna check if the matrix is singular before implementing calculation process.
        """
        N, D = points.shape
        diff = points - mu_i  
        
        try:
            # get inverse & determinant
            sigma_inv = np.linalg.inv(sigma_i)
            det_sigma = np.linalg.det(sigma_i)
        except LinAlgError:
            # singular matrix handling
            sigma_inv = np.linalg.inv(sigma_i + SIGMA_CONST * np.eye(D))
            det_sigma = np.linalg.det(sigma_i + SIGMA_CONST * np.eye(D))
        
        diff_sigma_inv = diff @ sigma_inv 
        quadratic = np.sum(diff_sigma_inv * diff, axis=1) 
        norm_const = 1.0 / ((2 * np.pi) ** (D / 2) * np.sqrt(det_sigma))
        normal_pdf = norm_const * np.exp(-0.5 * quadratic)
        
        return normal_pdf


    def create_pi(self):
        """
        Initialize the prior probabilities
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        pi = np.ones(self.K) / self.K
        return pi
        

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        indices = np.random.choice(self.N, size=self.K, replace=True)
        mu = self.points[indices]
        return mu


    def create_mu_kmeans(self, kmeans_max_iters=1000, kmeans_rel_tol=1e-05):
        """
        Intialize centers for each gaussian using your KMeans implementation from Q1
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        kmeans = KMeans(self.points, self.K, kmeans_max_iters)
        mu = kmeans()[1]  # get centroids as mu
        return mu


    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        sigma = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            sigma[k] = np.eye(self.D)
        return sigma


    def _init_components(self, kmeans_init=False, **kwargs):
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
            Note: If you have used random initialization for centers, invoke create_mu(), else if you have used
            your KMeans implementation from Q1 for center initialization, invoke create_mu_kmeans().
        """
        np.random.seed(5)
        pi = self.create_pi()
        
        if kmeans_init:
            mu = self.create_mu_kmeans()
        else:
            mu = self.create_mu()
            
        sigma = self.create_sigma()
        return pi, mu, sigma


    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        ll = np.zeros((self.N, self.K))
        
        for k in range(self.K):
            if full_matrix:
                pdf_values = self.multinormalPDF(self.points, mu[k], sigma[k])
            else:
                pdf_values = self.normalPDF(self.points, mu[k], sigma[k])
            
            ll[:, k] = np.log(pi[k] + LOG_CONST) + np.log(pdf_values + LOG_CONST) #adding small const. to prevent log(0)
        return ll


    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        log_likelihood = self._ll_joint(pi, mu, sigma, full_matrix)
        tau = self.softmax(log_likelihood)
        return tau


    def _M_step(self, tau, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        N_k = np.sum(tau, axis=0)  
        pi = N_k / self.N #update pi
        mu = np.zeros((self.K, self.D)) #update mu

        for k in range(self.K):
            if N_k[k] > 0:
                mu[k] = np.sum(tau[:, k:k+1] * self.points, axis=0) / N_k[k]
        
        sigma = np.zeros((self.K, self.D, self.D)) #update sigma

        for k in range(self.K):
            if N_k[k] > 0:
                diff = self.points - mu[k] 
                weighted_diff = tau[:, k:k+1] * diff 
                sigma[k] = (weighted_diff.T @ diff) / N_k[k]
                
                if not full_matrix:
                    #keep only diagonal elements
                    sigma[k] = np.diag(np.diag(sigma[k]))
        
        return pi, mu, sigma


    def __call__(
        self, full_matrix=FULL_MATRIX, kmeans_init=False, rel_tol=1e-16, **kwargs
    ):
        """
        Args:
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(kmeans_init, **kwargs)
        pbar = tqdm(range(self.max_iters))
        prev_loss = None
        for it in pbar:
            tau = self._E_step(pi, mu, sigma, full_matrix)
            pi, mu, sigma = self._M_step(tau, full_matrix)
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description("iter %d, loss: %.4f" % (it, loss))
            self.num_iters += 1
        return tau, (pi, mu, sigma)


def cluster_pixels_gmm(image, K, max_iters=10, full_matrix=True):
    """
    Clusters pixels in the input image

    Each pixel can be considered as a separate data point (of length 3),
    which you can then cluster using GMM. Then, process the outputs into
    the shape of the original image, where each pixel is its most likely value.

    Args:
        image: input image of shape(H, W, 3)
        K: number of components
        max_iters: maximum number of iterations in GMM. Default is 10
        full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
    Return:
        clustered_img: image of shape(H, W, 3) after pixel clustering

    Hints:
        What do mu and tau represent?
    """
    H, W, Z = image.shape
    pixels = image.reshape(-1, 3)
    
    gmm = GMM(pixels, K, max_iters)
    tau, (pi, mu, sigma) = gmm(full_matrix=full_matrix)
    
    cluster_assignments = np.argmax(tau, axis=1) 
    clustered_pixels = mu[cluster_assignments] 
    
    img_min = np.min(image)
    img_max = np.max(image)
    
    clustered_pixels = np.clip(clustered_pixels, img_min, img_max)
    clustered_img = clustered_pixels.reshape(H, W, Z)
    clustered_img = clustered_img.astype(image.dtype)
    
    return clustered_img


def density(points, pi, mu, sigma, gmm):
    """
    Evaluate the density at each point on the grid.
    Args:
        points: (N, 2) numpy array containing the coordinates of the points that make up the grid.
        pi: (K,) numpy array containing the mixture coefficients for each class
        mu: (K, D) numpy array containing the means of each cluster
        sigma: (K, D, D) numpy array containing the covariance matrixes of each cluster
        gmm: an instance of the GMM model

    Return:
        densities: (N, ) numpy array containing densities at each point on the grid

    HINT: You should be using the formula given in the hints.
    """
    N = points.shape[0]
    K = len(pi)
    densities = np.zeros(N)
    
    for k in range(K):
        pdf_values = gmm.multinormalPDF(points, mu[k], sigma[k])
        densities += pi[k] * pdf_values
    
    return densities


def rejection_sample(xmin, xmax, ymin, ymax, pi, mu, sigma, gmm, dmax=1, M=0.1):
    """
    Performs rejection sampling. Keep sampling datapoints until d <= f(x, y) / M
    Args:
        xmin: lower bound on x values
        xmax: upper bound on x values
        ymin: lower bound on y values
        ymax: upper bound on y values
        gmm: an instance of the GMM model
        dmax: the upper bound on d
        M: scale_factor. can be used to control the fraction of samples that are rejected

    Return:
        x, y: the coordinates of the sampled datapoint

    HINT: Refer to the links in the hints
    """
    while True:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        point = np.array([[x, y]])

        point_density = density(point, pi, mu, sigma, gmm)[0]
        random_val = np.random.uniform(0, dmax)
        if random_val <= point_density / M:
            return x, y


