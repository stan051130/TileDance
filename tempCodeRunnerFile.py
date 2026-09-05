    env.reset()
        env.draw_tile()
        env.sort_hand()

        done = 0
        step_count = 0

        while done == 0:

            state = env.get_state()

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            q_values = network(state_tensor)

            if random.random() < epsilon:
                action = random.randint(
                    0,
                    len(env.hand) - 1
                )
            else:
                action = torch.argmax(q_values).item()

            current_q = q_values[action]

            next_hand, reward, done = env.step(action)

            next_state = env.get_state()

            next_state_tensor = torch.tensor(
                next_state,
                dtype=torch.float32
            )

            with torch.no_grad():

                if done == 1:
                    target_q = torch.tensor(
                        float(reward),
                        dtype=torch.float32
                    )

                else:
                    next_q_values = network(next_state_tensor)

                    max_next_q = torch.max(next_q_values)

                    target_q = reward + gamma * max_next_q

            loss = loss_function(
                current_q,
                target_q
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_count += 1

        # ----- episode finished -----

        if reward == 1:
            wins += 1

        if (episode + 1) % 100 == 0:
            print(
                "Episode:",
                episode + 1,
                "Wins in last 100:",
                wins
            )

            wins = 0