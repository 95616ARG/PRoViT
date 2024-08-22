data/ILSVRC2012: data/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz data/ILSVRC2012/ILSVRC2012_img_val.tar

url_ILSVRC2012_devkit_t12=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
sha256_ILSVRC2012_devkit_t12=b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953
data/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz:
	mkdir -p $(@D)
	wget ${url_ILSVRC2012_devkit_t12} -P $(@D) \
		&& echo "${sha256_ILSVRC2012_devkit_t12} $@" | sha256sum --check \
		&& touch $@ \
	|| ( \
		echo "The downloaded $@ does not match the known sha256 ${sha256_ILSVRC2012_devkit_t12}." && \
		rm -f $@ \
	)

url_ILSVRC2012_img_val=http://academictorrents.com/download/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent
data/ILSVRC2012/ILSVRC2012_img_val.tar:
	@if test -f $@; then \
		echo "Found \"$@\"."; \
	else \
		echo "Please download the ImageNet validation set \"$@\" (6.3G) from \"${url_ILSVRC2012_img_val}\"."; \
		exit 1; \
	fi

datasets-imagenet: data/ILSVRC2012