
namespace ippl {
Vector<Vector<unsigned, 3>, 8> getLocalNodes() {
    return {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
}

}  // namespace ippl