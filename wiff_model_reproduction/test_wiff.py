"""Numerical checks against the archived representations and literal MATLAB logic."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from wiff import DenseNetwork, SPWIFF, parameters, peakfinder, histogram_metrics, predict_gated


def literal_peaks(y, spacing):
    maxima = [i for i in range(len(y)) if y[i] >= max(y[max(0,i-spacing):i+spacing+1])]
    minima = [i for i in range(len(y)) if y[i] <= min(y[max(0,i-spacing):i+spacing+1])]
    for left,right in zip(minima[:-1],minima[1:]):
        if right>left+1 and not any(left<i<right for i in maxima):
            maxima=sorted(maxima+[left+1+int(np.argmax(y[left+1:right]))])
    for left,right in zip(maxima[:-1],maxima[1:]):
        if right>left+1 and not any(left<i<right for i in minima):
            minima=sorted(minima+[left+1+int(np.argmin(y[left+1:right]))])
    return np.asarray(maxima),np.asarray(minima)


@pytest.mark.parametrize("seed",range(5))
def test_peakfinder_matches_literal_matlab(seed):
    y=np.random.default_rng(seed).normal(size=400)
    for actual,expected in zip(peakfinder(y,10),literal_peaks(y,10)):
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize("kind",["histogram","classifier"])
def test_text_export_matches_independent_hdf5_weights(kind):
    inputs=np.random.default_rng(17).uniform(0,1,(31,27))
    inputs[:,25]*=3
    with threadpool_limits(limits=1):
        a=DenseNetwork(kind,"text").predict(inputs)
        b=DenseNetwork(kind,"hdf5").predict(inputs)
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)


def test_batch_softmax_has_independent_rows():
    net=DenseNetwork()
    x=np.random.default_rng(9).uniform(0,3,(7,27))
    with threadpool_limits(limits=1):
        batch=net.predict(x)
        single=np.concatenate([net.predict(row) for row in x])
    np.testing.assert_allclose(batch,single,atol=1e-12)
    np.testing.assert_allclose(batch.sum(axis=1),1,atol=1e-14)


def test_zero_spectrum_does_not_fracture_and_gate_returns_zero():
    with threadpool_limits(limits=1):
        out=SPWIFF().run(np.zeros(25),1)
        prediction,probability,run=predict_gated(np.r_[np.zeros(25),1,.5])
    assert out.iterations==2 and out.converged
    assert not out.histogram.any() and not prediction.any() and not run.any()


def test_metric_is_weighted_absolute_error_not_sum_squared_error():
    true=np.array([[.4,.6]])
    pred=np.array([[.5,.5]])
    values=histogram_metrics(true,pred,centers=[10,100])
    np.testing.assert_allclose(values['rse_m'],[11])
    np.testing.assert_allclose(values['sse'],[11/64])


def test_seeded_sp_repeats_and_retains_probability_mass():
    f,_,_,_=parameters()
    spectrum=np.exp(-((f-.1)/.035)**2)
    with threadpool_limits(limits=1):
        kernel=SPWIFF(domain=1000)
        a=kernel.run(spectrum,1,seed=9,max_iterations=5)
        b=kernel.run(spectrum,1,seed=9,max_iterations=5)
    np.testing.assert_array_equal(a.histogram,b.histogram)
    np.testing.assert_allclose(a.histogram.sum(),1)


def test_fixed_phase_curvature_histogram_matches_literal_reference():
    kernel=SPWIFF(domain=1000)
    f,df,edges,centers=parameters()
    spectrum=np.exp(-((f-.12)/.045)**2)
    phase=np.linspace(0,2*np.pi,25)
    thickness=1.3
    wavelength=9.80616/(2*np.pi*f**2)
    eta=np.sum(np.sqrt(2*spectrum*df)[:,None]*np.cos(2*np.pi/wavelength[:,None]*kernel.x+phase[:,None]),axis=0)
    maxima,minima=literal_peaks(eta,10)
    locs=np.unique(np.r_[maxima,minima]);positions=kernel.x[locs]
    fracture_points=[]
    for i in range(1,len(locs)-1):
        left,middle,right=locs[i-1:i+2]
        dx1=kernel.x[middle]-kernel.x[left];dx2=kernel.x[right]-kernel.x[middle]
        strain=abs(thickness*(eta[left]*dx2-eta[middle]*(dx1+dx2)+eta[right]*dx1)/(dx1*dx2*(dx1+dx2)))
        if strain>3e-5:fracture_points.append(kernel.x[middle])
    radii=np.diff(fracture_points)/2
    counts=np.histogram(radii,edges)[0];counts[-1]+=sum(radii>edges[-1])
    weighted=counts*centers;expected=weighted/weighted.sum()
    with threadpool_limits(limits=1):actual=kernel.run(spectrum,thickness,max_iterations=1,phases=phase)
    np.testing.assert_array_equal(actual.histogram,expected)


def test_reject_nonfinite_network_input():
    with pytest.raises(ValueError):DenseNetwork().predict(np.full(27,np.nan))


def test_pinned_upstream_files_match_provenance():
    from wiff import ROOT,VENDOR
    import json,hashlib
    manifest=json.loads((ROOT/'vendor/manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((VENDOR/item['path']).read_bytes()).hexdigest()==item['sha256']
