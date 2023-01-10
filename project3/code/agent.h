/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include <functional>
#include <thread>
#include "board.h"
#include "action.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		std::random_device rd;
		engine.seed(rd());
	}
	virtual ~random_agent() {}

protected:
	std::mt19937 engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class random_player : public random_agent {
public:
	random_player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

protected:
	std::vector<action::place> space;
	board::piece_type who;
};

/**
 * player for both side
 * MCTS: perform N cycles and take the best action by visit count
 * random: put a legal piece randomly
 */
class player : public random_player {
public:
	player(const std::string& args = "") : random_player("name=player role=unknown search=MCTS simulation=0 " + args) {
		if (property("search") != "MCTS")
			throw std::invalid_argument("invalid search: " + property("search"));
		if (meta.find("timeout") != meta.end())
			throw std::invalid_argument("invalid option: timeout");
	}

	virtual action take_action(const board& state) {
		size_t N = meta["simulation"];
		int parallel = meta["parallel"];
		int T = meta["T"];
		if (N){
			std::vector<std::thread> threads;
			std::vector<std::vector<int>> visits(parallel, std::vector<int>(81, 0));
			std::vector<node> nodes;
			for(int i=0;i<parallel;i++){
				nodes.push_back(node(state));
			}
			for(int i=0;i<parallel;i++){
				threads.push_back(std::thread(&node::run_mcts, &nodes[i], N, T, std::ref(visits[i]), std::ref(engine)));
			}
			for(int i=0;i<parallel;i++){
				threads[i].join();
			}

			std::vector<int> sum(81, 0);
			for(auto vis: visits){
				for(int i=0;i<81;i++){
					sum[i] += vis[i];
				}
			}
			action::place best_action = action();
			int max_visit = 0;
			for(int i=0;i<81;i++){
				// std::cout << sum[i] << " ";
				if(sum[i] >= max_visit){
					max_visit = sum[i];
					best_action = action::place(i, who);
				}
			}
			// std::cout << std::endl;

			return best_action;

		}
		return random_player::take_action(state);
	}

	class node : board {
	public:
		node(const board state, node* parent = nullptr) : board(state),
			win(0), visit(0), child(), parent(parent) {}

		/**
		 * run MCTS for N cycles and retrieve the best action
		 */
		void run_mcts(size_t N, int T, std::vector<int>& visits, std::mt19937& engine) {
			const auto threshold = std::chrono::milliseconds(T);
			auto start_time = std::chrono::high_resolution_clock::now();
			for (size_t i = 0; i < N && (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) < threshold); i++) {
				std::vector<node*> path = select();
				node* leaf = path.back()->expand(engine);
				if (leaf != path.back()) path.push_back(leaf);
				update(path, leaf->simulate(engine));
			}
			take_action(visits);
		}

	protected:

		/**
		 * select from the current node to a leaf node by UCB and return all of them
		 * a leaf node can be either a node that is not fully expanded or a terminal node
		 */
		std::vector<node*> select() {
			std::vector<node*> path = { this };
			for (node* ndptr = this; ndptr->is_selectable(); path.push_back(ndptr)) {
				ndptr = &*std::max_element(ndptr->child.begin(), ndptr->child.end(),
						[=](const node& lhs, const node& rhs) { return lhs.ucb_score() < rhs.ucb_score(); });
			}
			return path;
		}

		/**
		 * expand the current node and return the newly expanded child node
		 * if the current node has no unexpanded move, it returns itself
		 */
		node* expand(std::mt19937& engine) {
			board child_state = *this;
			std::vector<int> moves = all_moves(engine);
			auto expanded_move = std::find_if(moves.begin(), moves.end(), [&](int move) {
				// check whether it is an unexpanded legal move
				bool is_expanded = std::find_if(child.begin(), child.end(),
						[&](const node& node) { return node.info().last_move_index == move; }) != child.end();
				return is_expanded == false && child_state.place(move) == board::legal;
			});
			if (expanded_move == moves.end()) return this; // already terminal
			child.emplace_back(child_state, this);
			return &child.back();
		}

		/**
		 * simulate the current node and return the winner
		 */
		unsigned simulate(std::mt19937& engine) {
			board rollout = *this;
			std::vector<int> moves = all_moves(engine);
			while (std::find_if(moves.begin(), moves.end(),
					[&](int move) { return rollout.place(move) == board::legal; }) != moves.end());
			return (rollout.info().who_take_turns == board::white) ? board::black : board::white;
		}

		/**
		 * update statistics for all nodes saved in the path
		 */
		void update(std::vector<node*>& path, unsigned winner) {
			for (node* ndptr : path) {
				ndptr->win += (winner == info().who_take_turns) ? 1 : 0;
				ndptr->visit += 1;
			}
		}

		/**
		 * pick the best action by visit counts
		 */
		void take_action(std::vector<int>& visits) {
			// auto best = std::max_element(child.begin(), child.end(),
			// 		[](const node& lhs, const node& rhs) { return lhs.visit < rhs.visit; });
			// if (best == child.end()) return action(); // no legal move
			// return action::place(best->info().last_move_index, info().who_take_turns);
			for(auto i: child){
				visits[i.info().last_move_index] += i.visit;
			}
		}

	private:

		/**
		 * check whether this node is a fully-expanded non-terminal node
		 */
		bool is_selectable() const {
			size_t num_moves = 0;
			for (int move = 0; move < 81; move++)
				if (board(*this).place(move) == board::legal)
					num_moves++;
			return child.size() == num_moves && num_moves > 0;
		}

		/**
		 * get the ucb score of this node
		 */
		float ucb_score(float c = 0.75) const {
			float exploit = float(win) / visit;
			float explore = std::sqrt(std::log(parent->visit) / visit);
			return exploit + c * explore;
		}

		/**
		 * get all moves in shuffled order
		 */
		std::vector<int> all_moves(std::mt19937& engine) const {
			std::vector<int> moves;
			for (int move = 0; move < 81; move++) moves.push_back(move);
			std::shuffle(moves.begin(), moves.end(), engine);
			return moves;
		}

	public:
		size_t win, visit;
		std::vector<node> child;
		node* parent;
	};
};
