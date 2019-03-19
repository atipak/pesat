with open("/home/patik/Diplomka/dp_ws/src/py_scripts/input", "r") as file:
    string = file.readlines()
    st = ""
    for s in string:
        st += s
    print(st.format("10"))