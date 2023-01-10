import numpy as np
from numpy.random import rand, randn


class InitialCondition:
    
    def __init__(self, n_particles=100, x_range=(-40, 40), y_range=(-40, 40)):
        self.n = n_particles
        self.xr = x_range
        self.yr = y_range

    @property
    def distx(self):
        return self.xr[1] - self.xr[0]
    
    @property
    def disty(self):
        return self.yr[1] - self.yr[0]

    @property
    def circular(self):
        halfdiam = 0.5 * self.distx
        r = 0.3*halfdiam + 0.3*halfdiam * rand(self.n)
        ang = 2*np.pi*rand(self.n)  # np.linspace(0, 2*np.pi, self.n)

        X = np.array([r * np.cos(ang), r*np.sin(ang)]).transpose()
        norm = np.sqrt(np.sum(X**2, 1))
        V = np.stack((-X[:, 1], X[:, 0]), axis=1) / np.stack((norm, norm), axis=1)
        return X, V

    @property
    def square(self):
        X = np.array([self.xr[0], self.yr[0]]) + rand(self.n, 2) * np.array([self.distx, self.disty])
        v = np.array([1, 1])
        V = np.array([v]*X.shape[0])
        return X, V

    @property
    def nospeed(self):
        X, V = self.square
        return X, np.zeros(V.shape)

    @property
    def random_speed(self):
        X, V = self.square
        return X, 10. * (rand(*V.shape)-0.5)



def linear_movement(X, V):
    return V, np.zeros(V.shape)


def rep_attr_potential(Ca=20.0, Cr=50.0, la=100, lr=2):
    def force(r):
        return Ca/la*np.exp(-r/la)-Cr/lr*np.exp(-r/lr)
    return force


def rep_attr_rhs(X, V, alpha=0.07, beta=0.05):

    force = rep_attr_potential()
    x1, x2 = np.meshgrid(X[:, 0], X[:, 0])
    distx1 = x1 - x2
    x1, x2 = np.meshgrid(X[:, 1], X[:, 1])
    distx2 = x1 - x2
    normdi = np.sqrt(distx1**2 + distx2**2)
    
    rhsv = V*(alpha - beta * (V**2 + np.flip(V, axis=1)**2)) - np.stack((
        np.sum(force(normdi)/((np.abs(normdi-0)<1e-14).astype(int) +normdi)*distx1, axis=0),
        np.sum(force(normdi)/((np.abs(normdi-0)<1e-14).astype(int) +normdi)*distx2, axis=0)
    )).T/X.shape[0]
    
    return V, rhsv

class Model:

    def __init__(self, X, V, rhs=linear_movement):
        self.X = X
        self.V = V
        self.rhs = rhs

    def evolve(self, time_step, update=True):
        rhsx, rhsv = self.rhs(self.X, self.V)
        X_half = self.X + .5 * time_step * rhsx
        V_half = self.V + .5 * time_step * rhsv

        rhsx, rhsv = self.rhs(X_half, V_half)
        X = self.X + time_step * rhsx
        V = self.V + time_step * rhsv

        if update:
            self.X = X
            self.V = V
        return self

    def cds_dict(self):
        return dict(
            x1=self.X[:, 0],
            x2=self.X[:, 1],
            x1s=list(np.stack((self.X[:, 0], self.X[:, 0] + self.V[:, 0]), axis=1)),
            x2s=list(np.stack((self.X[:, 1], self.X[:, 1] + self.V[:, 1]), axis=1))
        )