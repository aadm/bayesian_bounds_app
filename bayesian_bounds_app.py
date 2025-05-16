# bayesian_bounds_app
#--------------------------------------------------------------------
#
# Demo app showing the bayesian space bounds constrained by fixed
# sensitivity and specificity parameters.
# 
# to run: 
#
# $ streamlit run bayesian_bounds_app.py
#--------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

#--------------------------------------------------------------------

def noise_uniform(center, halfrange, samples):
    rng = np.random.default_rng()
    lower_bound = center-halfrange
    if lower_bound<0.1:
        lower_bound=0.1
    upper_bound = center+halfrange
    if upper_bound>1.0:
        upper_bound=1.0
    X = rng.uniform(low=lower_bound, high=upper_bound, size=samples)
    return X


def noise_gaussian(center, halfrange, samples):
    from scipy.stats import truncnorm
    a, b = 0.5, 1.0
    sd = halfrange/3
    X = truncnorm((a-center)/sd, (b-center)/sd, loc=center, scale=sd)
    return X.rvs(samples)


def marginal_probability(prior, sens, spec):
    return sens*prior + (1-spec)*(1-prior)


def posterior_true_test(prior, sens, spec):
    '''
    Calculate posterior probability P(H|A)
    i.e. prob of event H being true if A is true
    '''
    return (sens*prior) / marginal_probability(prior, sens, spec)


def simulation_posterior(pos, sens, spec, noise_type, noise_var, samples):

    prior_lab = 'Initial POS'
    post_lab = 'Updated POS'

    # do random things
    howmany = ['low', 'medium', 'high']
    howmany_number = [50, 500, 5000]
    ns = dict(zip(howmany, howmany_number))

    alpha_values = [0.8, 0.4, 0.1]
    alf = dict(zip(howmany, alpha_values))

    # add noise
    if noise_type == 'uniform':
        sens0 = noise_uniform(sens, noise_var, ns[samples])
        spec0 = noise_uniform(spec, noise_var, ns[samples])
    else:
        sens0 = noise_gaussian(sens, noise_var, ns[samples])
        spec0 = noise_gaussian(spec, noise_var, ns[samples])

    # draw random samples to build prior POS distribution
    rng = np.random.default_rng()
    prior = rng.normal(loc=np.mean(pos), scale=np.std(pos), size=ns[samples])

    # set plot options
    opt = dict(marker='o', markersize=5, mec='none', ls='none', alpha=alf[samples], color='red')

    # make figure
    # fig, ax = plt.subplots(figsize=(8,4), ncols=2, constrained_layout=True)
    fig, ax = plt.subplots()
    ax.plot(prior, posterior_true_test(prior, sens0, spec0), **opt)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()
    ax.set_xlabel(prior_lab)
    ax.set_ylabel(post_lab)
    ax.tick_params(axis='both', which='major')
    ax.set_aspect(1)
    ax.set_title(f'Sensitivity={sens:.2f}, Specificity={spec:.2f}')

    # ax[1].hist(prior, color='magenta', alpha=0.5, bins=50)
    # ax[1].hist(posterior_true_test(prior, sens0, spec0), color='blue', alpha=0.5, bins=50)
    # ax[1].legend(['Initial POS', 'Updated POS'])
    # ax[1].set_xlim(0, 1)
    # fig.suptitle(f'Sensitivity={sens:.2f}, Specificity={spec:.2f}')

    return fig


#--------------------------------------------------------------------

st.set_page_config(page_title='Bayesian Bounds', page_icon=':elephant:')

cols = st.columns([0.6, 0.2, 0.2], gap='small', vertical_alignment="bottom")
with cols[0]:
    st.title('Bayesian Bounds')
with cols[2]:
    st.markdown('_aadm, 2023-2025_')
st.divider()


st.sidebar.header("Bayesian parameters")
wpos = st.sidebar.slider('Prior POS or P(H)', 0.05, 1.0, (0.4, 0.8), 0.05)
wsens = st.sidebar.slider('Sensitivity', 0.0, 1.0, 0.5, 0.01)
wspec = st.sidebar.slider('Specificity', 0.0, 1.0, 0.5, 0.01)

st.sidebar.header("Simulation parameters")
# wnoise_type = st.sidebar.radio('Type of Noise', ['uniform', 'gaussian'], index=0)
wnoise_type = 'uniform'
wnoise = st.sidebar.slider('Max noise', min_value=0.001, max_value=0.5, value=0.001, step=0.01)
wsamp = st.sidebar.radio('N. of samples', ['low', 'medium', 'high'], index=1)

# if st.button("Run Simulation"):
fig = simulation_posterior(wpos, wsens, wspec, wnoise_type, wnoise, wsamp)
st.pyplot(fig)
    # st.subheader("Interactive crossplot prior POS - final POS (with histograms)")
    # simulation_posterior_v2(wpos, wsens, wspec, wnoise_type, wnoise, wsamp)

# st.sidebar.header("VOI posterior probability")
# wpos2 = st.sidebar.slider('Prior POS or P(H)', 0.001, 1.0, (0.1, 0.8), 0.05)
# wsensg = st.sidebar.radio('Expectation False Negatives', ['n/a', 'many', 'some', 'few', 'rare'])
# wspecg = st.sidebar.radio('Expectation False Positives', ['n/a', 'many', 'some', 'few', 'rare'])

# if st.sidebar.button("Plot Posterior Grade"):
#     st.subheader("VOI posterior probability")
#     plot_posterior_grade(wpos2, wsensg, wspecg)
