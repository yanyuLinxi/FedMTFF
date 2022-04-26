class batch:
    def __init__(self):
        self.a="a"
        print("has a", hasattr(self, "a"))
        print("has b", hasattr(self, "b"))
        
    def generator(self):
        for i in range(10):
            yield i
        

b = batch()
print(hasattr(b, "a"))
# for term in b.generator():
#     print(term)
