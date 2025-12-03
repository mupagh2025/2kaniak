from PyMPDATA import ScalarField, Solver, Stepper, VectorField, Options, boundary_conditions
import numpy as np
import scipy


class ShallowWaterEquationsIntegrator:
    def __init__(self, *, bathymetry: np.ndarray, h_initial: np.ndarray, options: Options = None):
        """ initializes the solvers for a given initial condition of `h` assuming zero momenta at t=0 """
        options = options or Options(nonoscillatory=True, infinite_gauge=True)
        X, Y, grid = 0, 1, h_initial.shape
        stepper = Stepper(options=options, grid=grid)
        kwargs = {
            'boundary_conditions': [boundary_conditions.Constant(value=0)] * len(grid),
            'halo': options.n_halo,
        }
        advectees = {
            "h": ScalarField(h_initial, **kwargs),
            "uh": ScalarField(np.zeros(grid), **kwargs),
            "vh": ScalarField(np.zeros(grid), **kwargs),
        }
        self.advector = VectorField((
                np.zeros((grid[X] + 1, grid[Y])),
                np.zeros((grid[X], grid[Y] + 1))
            ), **kwargs
        )
        self.solvers = { k: Solver(stepper, v, self.advector) for k, v in advectees.items() }
        self.bathymetry = bathymetry

    def __getitem__(self, key):
        """ returns `key` advectee field of the current solver state """
        return self.solvers[key].advectee.get()
    
    def _apply_half_rhs(self, *, key, axis, g_times_dt_over_dxy):
        """ applies half of the source term in the given direction """
        self[key][:] -= .5 * g_times_dt_over_dxy * self['h'] * np.gradient(self['h'] - self.bathymetry, axis=axis)

    def _update_courant_numbers(self, *, axis, key, mask, dt_over_dxy):
        """ computes the Courant number component from fluid column height and momenta fields """
        velocity = np.where(mask, np.nan, 0)
        momentum = self[key]
        np.divide(momentum, self['h'], where=mask, out=velocity)

        # using slices to ensure views (over copies)
        all = slice(None, None) 
        all_but_last = slice(None, -1)
        all_but_first_and_last = slice(1, -1)

        velocity_at_cell_boundaries = velocity[( 
            (all_but_last, all),
            (all, all_but_last),
        )[axis]] + np.diff(velocity, axis=axis) / 2 
        courant_number = self.advector.get_component(axis)[(
            (all_but_first_and_last, all),
            (all, all_but_first_and_last)
        )[axis]]
        courant_number[:] = velocity_at_cell_boundaries * dt_over_dxy[axis]
        assert np.amax(np.abs(courant_number)) <= 1

    def __call__(self, *, nt: int, g: float, dt_over_dxy: tuple, outfreq: int, eps: float=1e-7, bathymetryCallback=None):
        """ integrates `nt` timesteps and returns a dictionary of solver states recorded every `outfreq` step[s] """
        
        output = {k: [] for k in self.solvers.keys()}
        output['bathymetry'] = []
        for it in range(nt + 1):

            if bathymetryCallback is not None:
                bathymetryCallback(self.bathymetry, it)

            if it != 0:
                mask = self['h'] > eps
                for axis, key in enumerate(("uh", "vh")):
                    self._update_courant_numbers(axis=axis, key=key, mask=mask, dt_over_dxy=dt_over_dxy)
                self.solvers["h"].advance(n_steps=1)
                for axis, key in enumerate(("uh", "vh")):
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
                    self.solvers[key].advance(n_steps=1)
                    self._apply_half_rhs(key=key, axis=axis, g_times_dt_over_dxy=g * dt_over_dxy[axis])
            if it % outfreq == 0:
                output['bathymetry'].append(self.bathymetry.copy())
                for key in self.solvers.keys():
                    output[key].append(self[key].copy())
                    
        return output
    
class UpliftParams:
    def __init__(self):
        self.grid = 0, 0
        self.center = 0, 0
        self.sigmas = 0, 0
        self.magnitude = 0
        self.tStart = 0
        self.tEnd = 0

def createGaussianUplift(up: UpliftParams):
    x = np.arange(up.grid[0])
    y = np.arange(up.grid[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    x0, y0 = up.center
    sigX, sigY = up.sigmas

    shapeX = (X - x0)**2 / (2*sigX**2)
    shapeY = (Y - y0)**2 / (2*sigY**2)
    shape =  up.magnitude * np.exp(- shapeX - shapeY)

    steps = (up.tEnd - up.tStart) / up.dt
    shapeContribution = shape / steps

    def updateBathymetry(bathymetry, it):
        itStart = up.tStart / up.dt
        itEnd = up.tEnd / up.dt
        if itStart <= it < itEnd:
            bathymetry -= shapeContribution

    return updateBathymetry


def calculate_total_energy(output, dx, dy, rho=1000.0, g=9.81):
    ep_history = []
    ek_history = []
    total_history = []
    
    n_frames = len(output['h'])

    dxdy = dx * dy
    
    for i in range(n_frames):

        h = output['h'][i]
        bathymetry = output['bathymetry'][i]

        uh = output['uh'][i]
        vh = output['vh'][i]
        
        eta = h - bathymetry
        
        ep_step = 0.5 * rho * g * np.sum(eta**2) * dxdy

        ek_density = 0.5 * rho * (uh**2 + vh**2) / (h + 1e-6)
        ek_step = np.sum(ek_density) * dxdy

        ep_history.append(ep_step)
        ek_history.append(ek_step)
        total_history.append(ep_step + ek_step)
        
    return np.array(ep_history), np.array(ek_history), np.array(total_history)

def calculate_velocities(output, dt, dx, outfreq, g=9.81):
    """
    Zwraca prędkość wyznaczoną i teoretyczną
    """
    ### WYZNACZANIE PRĘDKOŚCI
    def linear_f(x, a, b):
        return a * x + b
    
    H = np.mean(output['h'][0])


    t_meas = [3.0, 3.5, 4.0, 4.5, 5.0]

    meas_idx = [int(tm / dt / outfreq) for tm in t_meas]

    positions = []

    for midx in meas_idx:
        psi = output['h'][midx] - output['bathymetry'][midx]
        psi = psi[:, 200]
        position_idx = np.argmax(psi[200:])
        position = (position_idx + 200) * dx
        positions.append(position)

    fit = scipy.optimize.curve_fit(linear_f, t_meas, positions)

    popt, pcov = fit

    vel_fit = popt[0]
    vel_theo = np.sqrt( g * (H ) ) * (1 + np.max(psi)/(2*H))

    return vel_fit, vel_theo