# check extra module installation
if ! python -c 'import causal_conv1d' > /dev/null; then
    echo "Warning: it seems that causal_conv1d is not installed." >&2
    echo "Warning: please install causal_conv1d as follows." >&2
    echo "Warning: cd ${MAIN_ROOT}/tools && make pip install causal-conv1d" >&2
    return 1
fi