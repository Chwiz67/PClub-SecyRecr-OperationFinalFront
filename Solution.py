from pred import pred

def solve(n, adj, s, d, cost, curr_cost, vis):
    if s == n - 1:
        cost.append(curr_cost)
        return

    for i in range(n):
        if adj[s][i] == 1 and d == 0 and not vis[i]:
            vis[i] = True
            solve(n, adj, i, 0, cost, curr_cost + 1, vis)
            vis[i] = False

        if adj[s][i] == 1 and d == 1 and not vis[i]:
            vis[i] = True
            solve(n, adj, i, 0, cost, curr_cost + n + 1, vis)
            vis[i] = False

        if adj[i][s] == 1 and d == 1 and not vis[i]:
            vis[i] = True
            solve(n, adj, i, 1, cost, curr_cost + 1, vis)
            vis[i] = False

        if adj[i][s] == 1 and d == 0 and not vis[i]:
            vis[i] = True
            solve(n, adj, i, 1, cost, curr_cost + n + 1, vis)
            vis[i] = False

if __name__ == "__main__":
    
    img_path = input("Enter the path to the image: ")
    n, adj = pred(img_path)

    cost = []
    vis = [False] * n
    vis[0] = True
    solve(n, adj, 0, 0, cost, 0, vis)

    if not cost:
        print(-1)
    else:
        print(min(cost))