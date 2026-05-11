#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace chrono;

vector<int> graph[50];
bool visited[50];

// Parallel BFS
void BFS(int start) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        cout << node << " ";

        #pragma omp parallel for
        for (int i = 0; i < graph[node].size(); i++) {
            int next = graph[node][i];

            if (!visited[next]) {
                #pragma omp critical
                {
                    if (!visited[next]) {
                        visited[next] = true;
                        q.push(next);
                    }
                }
            }
        }
    }
}

// DFS (fixed)
void DFS(int node) {
    if (visited[node])
        return;

    visited[node] = true;
    cout << node << " ";

    for (int i = 0; i < graph[node].size(); i++) {
        int next = graph[node][i];
        if (!visited[next]) {
            DFS(next);
        }
    }
}

int main() {
    int n, e, u, v, start;

    for (int i = 0; i < 50; i++)
        visited[i] = false;

    cout << "Enter number of nodes and edges: ";
    cin >> n >> e;

    cout << "Enter edges:\n";
    for (int i = 0; i < e; i++) {
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    cout << "Enter starting node: ";
    cin >> start;

    auto t1 = high_resolution_clock::now();
    cout << "\nBFS Traversal: ";
    BFS(start);
    auto t2 = high_resolution_clock::now();

    for (int i = 0; i < 50; i++)
        visited[i] = false;

    auto t3 = high_resolution_clock::now();
    cout << "\nDFS Traversal: ";
    DFS(start);
    auto t4 = high_resolution_clock::now();

    cout << "\n\nBFS Time: "
         << duration_cast<microseconds>(t2 - t1).count()
         << " microseconds";

    cout << "\nDFS Time: "
         << duration_cast<microseconds>(t4 - t3).count()
         << " microseconds";

    return 0;
}
