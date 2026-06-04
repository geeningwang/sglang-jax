import sys
sys.path.append("/home/jingnw_google_com/.local/lib/python3.9/site-packages")
try:
    from flax import nnx
    s = nnx.State({'a': 1})
    print(s.pop('a'))
except Exception as e:
    print(e)
