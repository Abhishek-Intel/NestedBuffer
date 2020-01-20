#include "CL/sycl.hpp"
#include <vector>
#include <utility>
#include <iostream>
#include <memory>

struct C {
    int m_cData;

    C() : m_cData(0) {}

    ~C() {
        std::cout << "C deallocating" << std::endl;
    }
};

struct B {
    int m_bData;
    std::shared_ptr<C> m_c;

    B() : m_bData(0), m_c(std::make_shared<C>()) {}

    ~B() {
        std::cout << "B deallocating" << std::endl;
    }
};

template<cl::sycl::access::target target>
struct BBuff {
    cl::sycl::buffer<B> m_bBuff;
    cl::sycl::buffer<C> m_cBuff;

    BBuff(const std::shared_ptr<B>& b) : m_bBuff(b, cl::sycl::range<1>(1)), m_cBuff(b->m_c, cl::sycl::range<1>(1)) {}

    /*~BBuff() {
        std::cout << "BBuff deallocating" << std::endl;
    }*/
};

template<cl::sycl::access::target target>
struct BView
{
    cl::sycl::accessor<B, 1, cl::sycl::access::mode::read_write, target, cl::sycl::access::placeholder::true_t> m_bDataAcc;

    cl::sycl::accessor<C, 1, cl::sycl::access::mode::read_write, target, cl::sycl::access::placeholder::true_t> m_cAcc;

    BBuff<target>* m_bBuff;

    BView(BBuff<target>* bBuff): m_bDataAcc(bBuff->m_bBuff), m_cAcc(bBuff->m_cBuff){ }

    void RequireForHandler(cl::sycl::handler& cgh) {
        cgh.require(m_bDataAcc);
        cgh.require(m_cAcc);
    }

    /*~BView()
    {
        std::cout << "BView deallocating" << std::endl;
    }*/
};

class init_first_block;

int main(int argc, char** argv)
{
    std::shared_ptr<B> b = std::make_shared<B>();
    try
    {
        cl::sycl::default_selector device_selector;
        cl::sycl::queue workQueue(device_selector);
        BBuff<cl::sycl::access::target::global_buffer> bGlobalBuff(b);
        BView<cl::sycl::access::target::global_buffer> bAccDevice(&bGlobalBuff);
        workQueue.submit([&bAccDevice](cl::sycl::handler &cgh) {
            bAccDevice.RequireForHandler(cgh);

            cgh.single_task<class init_first_block>([bAccDevice]() {
                bAccDevice.m_bDataAcc[0].m_bData = 1;
                bAccDevice.m_cAcc[0].m_cData = 3;
            });
        });

        workQueue.wait();
    }
    catch (...)
    {
        std::cout << "Failure running nested accessor test" << std::endl;
    }

    std::cout << "B data: " << b->m_bData << std::endl;
    std::cout << "C data: " << b->m_c->m_cData << std::endl;

    return 0;
}
