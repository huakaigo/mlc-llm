import tvm
from tvm import relay

class CustomTransformer(tvm.relay.ExprMutator):
    def visit_expr(self, expr):
        print('111')
        # 对所有节点进行预处理
        if isinstance(expr, relay.Constant):
            return relay.Constant(expr.data * 2)
        return super().visit_expr(expr)

    def visit_call(self, call):
        print('222')
        # 对函数调用节点进行处理
        if call.op.name == 'add':
            new_op = relay.op.get("subtract")
            new_args = [self.visit(arg) for arg in call.args]
            return relay.Call(new_op, new_args)
        return super().visit_call(call)

# 创建一个简单的表达式
x = relay.var('x')
y = relay.var('y')
add_expr = relay.add(x, y)
print(add_expr)
print('--------------------------------')

# 使用自定义的变换器进行变换
transformer = CustomTransformer()
result_expr = transformer.visit(add_expr)
func = relay.Function([x,y], add_expr)
print(type(result_expr))
print(result_expr)
r = func(tvm.relay.const(3) , tvm.relay.const(4))
print(r)
