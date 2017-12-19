import PyCall

if !isdefined(:gym)
    global const gym = PyCall.pywrap(PyCall.pyimport("gym"))
end
