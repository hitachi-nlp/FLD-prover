class Hoge:

    def echo_hoge(self):
        print('hoge')


def echo_hoge_with_arounds(func_org):

    def func_with_arounds(self, *args, **kwargs):
        print('I will echo hoge!')
        # mport pudb; pudb.set_trace()
        return func_org(self, *args, **kwargs)

    return func_with_arounds

Hoge.echo_hoge = echo_hoge_with_arounds(Hoge.echo_hoge)

hoge = Hoge()

# hoge.echo_hoge = echo_hoge_with_arounds(hoge.echo_hoge)

# hoge.echo_hoge = echo_hoge_with_arounds

hoge.echo_hoge()
