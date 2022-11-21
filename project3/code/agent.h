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
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
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

private:
	std::vector<action::place> space;
	board::piece_type who;
};

class mctsNode {
public:
	mctsNode(const board& state, const board::piece_type player) : parent(nullptr), chosen_action(), state(state), visit(0), win(0), expand_idx(0), end_state(false), player(player) {}
	mctsNode(mctsNode* parent, const action::place& action, const board& state, const bool end_state, const board::piece_type player) : parent(parent), chosen_action(action), 
	state(state), visit(0), win(0),	expand_idx(0), end_state(end_state), player(player) {}
	~mctsNode() {
		for (auto& child : children)
			delete child;
	}

	double UCB1() {
		if (visit == 0)
			return 1e9;
		return (double)win / (double)visit + sqrt((double)2 * (double)log(parent->visit) / double(visit));
	}

	mctsNode* parent;
	action::place chosen_action;
	board state;
	int visit;
	int win;
	int expand_idx;
	bool end_state;
	board::piece_type player;
	std::vector<mctsNode*> children;
};

class mctsPlayer : public random_agent {
public:
	mctsPlayer(const std::string& args="") : random_agent("name=mcts role=unknown " + args),
		black_space(board::size_x * board::size_y), white_space(board::size_x * board::size_y), who(board::empty){
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		root = new mctsNode(board(), (who == board::black ? board::white : board::black));
		real_root = root;
		for (size_t i = 0; i < black_space.size(); i++){
			black_space[i] = action::place(i, board::black);
			white_space[i] = action::place(i, board::white);
		}
		if (meta.find("T") != meta.end()){
			T = std::stoi(meta["T"]);
		}
	}

	~mctsPlayer() {
		delete real_root;
	}

	virtual action take_action(const board& state) {
		action action = mcts(T, state);
		return action;
	}

	action mcts(int num_of_simulations, const board& state){
		bool in_tree = false;
		for(auto i: root->children){
			if(i->state == state){
				root = i;
				root->parent = nullptr;
				in_tree = true;				
				break;
			}
		}

		if(!in_tree){
			delete real_root;
			root = new mctsNode(state, (who == board::black ? board::white : board::black));
			real_root = root;
		}

		while(num_of_simulations --){
			mctsNode* node = root;
			// std::cout << root -> visit << " " << root -> win << std::endl;
			// std::cout << num_of_simulations << std::endl;
			// selction
			while(node->children.size() != 0 && (int)node->children.size() == node->expand_idx){
				double max_UCB1 = -1e9;
				mctsNode* max_node = nullptr;
				for(auto& child : node->children){
					double UCB1 = child->UCB1();
					if(UCB1 > max_UCB1){
						max_UCB1 = UCB1;
						max_node = child;
					}
				}
				node = max_node;
			}

			// expansion
			if(node->end_state == false && (int)node->children.size() == 0){
				bool no_legal_move = true;
				if(node->player == board::black){
					std::shuffle(white_space.begin(), white_space.end(), engine);
					for (const action::place& move : white_space) {
						board after = node->state;
						if (move.apply(after) == board::legal){
							node->children.push_back(new mctsNode(node, move, after, false, board::white));
							no_legal_move = false;
						}
					}
				} else {
					std::shuffle(black_space.begin(), black_space.end(), engine);
					for (const action::place& move : black_space) {
						board after = node->state;
						if (move.apply(after) == board::legal){
							node->children.push_back(new mctsNode(node, move, after, false, board::black));
							no_legal_move = false;
						}
					}
				}
				if(no_legal_move){
					node->end_state = true;
				}
			}

			// simulation
			board::piece_type winner;
			if(node->end_state == false){
				int idx = node->expand_idx;
				node->expand_idx++;
				node = node->children[idx];
				board::piece_type player = node->player;
				board cur_state = node->state;
				while(true){
					bool no_legal_move = true;
					if(player == board::black){
						std::shuffle(white_space.begin(), white_space.end(), engine);
						for (const action::place& move : white_space) {
							board after = cur_state;
							if (move.apply(after) == board::legal){
								cur_state = after;
								no_legal_move = false;
								break;
							}
						}
					} else {
						std::shuffle(black_space.begin(), black_space.end(), engine);
						for (const action::place& move : black_space) {
							board after = cur_state;
							if (move.apply(after) == board::legal){
								cur_state = after;
								no_legal_move = false;
								break;
							}
						}
					}
					if(no_legal_move){
						winner = player;
						break;
					}
					player = (player == board::black ? board::white : board::black);
				}
			} else {
				winner = node->player;
			}
			
			// backpropagation
			while(node != nullptr){
				node->visit++;
				node->win += (winner == who);
				node = node->parent;
			}
		}

		double max_UCB1 = -1e9;
		mctsNode* max_node = nullptr;
		for(auto& child : root->children){
			double UCB1 = (double)child->win / (double)child->visit;
			if(UCB1 > max_UCB1){
				max_UCB1 = UCB1;
				max_node = child;
			}
		}
		if(max_node == nullptr){
			return action();
		}
		action::place action = max_node->chosen_action;
		root = max_node;
		return action;
	}



	int T = 100;
	std::vector<action::place> black_space;
	std::vector<action::place> white_space;
	board::piece_type who;
	mctsNode* root;
	mctsNode* real_root;
	std::string white_args;
};