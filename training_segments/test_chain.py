from pylab import *
import utilities

import hmm_chain
import cPickle
bumpy = cPickle.load(open("/home/pezzotto/log/bigbags/bag_files/databases/bumpy.pkl"))
pdc = bumpy['SLIDE_5CM']['pdc']; 
splits = [len(d) for d in pdc]
hmm = hmm_chain.HMMChain(data_splits=splits, n_pca_components=1, 
                         resampling_size=50, 
                         n_discretization_symbols=5)
hmm.update_splits(pdc)
print hmm.pipeline
pca = hmm.pca
pca.fit(vstack(pdc))
Xt = hmm.splitter.transform(pca.transform(hmm.combiner.transform(pdc)))
Xt = hmm.resample.fit_transform(Xt)
Xt = hmm.combiner.transform(Xt)
hmm.discretizer.fit(Xt)
Xt = hmm.discretizer.transform(Xt)
Xt = hmm.splitter2.transform(Xt)
hmm.hmm.fit(Xt)

print "Score: ", hmm.score(pdc)

print "Done"