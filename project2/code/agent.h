/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
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
#include <cassert>
#include "board.h"
#include "action.h"
#include "weight.h"

using namespace std;

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
 * base agent for agents with weight tables and a learning rate
 */

class state {
public:
	board cur, next;
	float reward;
	state(const board& _cur, const board& _next, float _reward){
		cur = _cur;
		next = _next;
		reward = _reward;
	}
};

class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}


	float board_value(const board& b){
		 // 0 ~ 3 are four rows, 4 ~ 7 are four columns.
		float value = 0;
		for(int i=0;i<4;i++){
			int tuple = 0;
			for(int j=0;j<4;j++){
				tuple <<= 4;
				tuple ^= b[i][j];
			}
			value += net[i][tuple];
		}
		for(int i=4;i<8;i++){
			int tuple = 0;
			for(int j=0;j<4;j++){
				tuple <<= 4;
				tuple ^= b[j][i-4];
			}
			value += net[i][tuple];
		}

		return value;		
	}

	void update_net(const board& b, float delta){
		for(int i=0;i<4;i++){
			int tuple = 0;
			for(int j=0;j<4;j++){
				tuple <<= 4;
				tuple ^= b[i][j];
			}
			net[i][tuple] += delta;
		}
		for(int i=4;i<8;i++){
			int tuple = 0;
			for(int j=0;j<4;j++){
				tuple <<= 4;
				tuple ^= b[j][i-4];
			}
			net[i][tuple] += delta;
		}
	}
	
	virtual void open_episode(const std::string& flag = "") {
		stats.clear();
	}

	virtual void close_episode(const std::string& flag = "") {
		int sz = stats.size();
		float delta = -1.0f * alpha * board_value(stats[sz-1].next);
		update_net(stats[sz-1].next, delta);
		for(int i=sz-1;i>=1;i--){
			float delta = alpha * (stats[i].reward + board_value(stats[i].next) - board_value(stats[i-1].next));
			update_net(stats[i-1].next, delta);
		}
	}


	virtual action take_action(const board& before) {
		float max_value = std::numeric_limits<float>::min();
		float max_reward = -1;
		int best_action = -1;

		for(auto i: {0, 1, 2, 3}){
			board after(before);
			board::reward reward = after.slide(i);
			float total = reward + board_value(after);
			if(reward != -1 and (total > max_value or max_value == std::numeric_limits<float>::min())){
				max_reward = reward;
				max_value = total;
				best_action = i;
			}
		}
		if(best_action != -1){
			board after(before);
			after.slide(best_action);
			stats.emplace_back(before, after, max_reward);
			return action::slide(best_action);
		} else {
			return action();
		}
	}

	virtual bool check_for_win(const board& b) {
		return false;
	}

	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	std::vector<state> stats;
	float alpha;
};


class iso_weight_agent : public agent {
public:
	iso_weight_agent(const std::string& args = "") : agent(args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
		
		build_iso_tuples();		
	}
	
	vector<vector<int>> tuples = {
		{0, 1, 2, 4, 5,  6},
		{1, 2, 5, 6, 9, 13},
		{0, 1, 2, 3, 4,  5},
		{0, 1, 5, 6, 7, 10}
	};

	vector<vector<int>> iso_tuples;
	
	void build_iso_tuples(){
		board b;
		for(int i=0;i<16;i++) b(i) = i;
		for(auto tuple: tuples){
			board tmp(b);
			for(int r=0;r<4;r++){
				vector<int> tmp_idx;
				for(auto idx: tuple){
					tmp_idx.push_back(tmp(idx));
				}
				iso_tuples.push_back(tmp_idx);
				tmp.rotate(1);
			}
			tmp.reflect_vertical();
			for(int r=0;r<4;r++){
				vector<int> tmp_idx;
				for(auto idx: tuple){
					tmp_idx.push_back(tmp(idx));
				}
				iso_tuples.push_back(tmp_idx);
				tmp.rotate(1);
			}
		}
	}

	float board_value(const board& b){
		float value = 0;
		
		// rotate
		int tuple_idx = 0;
		for(auto tuple: iso_tuples){
			int feature = 0;
			for(auto idx: tuple){
				feature <<= 4;
				feature ^= b(idx);
			}
			value += net[tuple_idx++/8][feature];
		}

		return value;
	}

	void update_net(const board& b, float delta){
		// rotate
		int tuple_idx = 0;
		for(auto tuple: iso_tuples){
			int feature = 0;
			for(auto idx: tuple){
				feature <<= 4;
				feature ^= b(idx);
			}
			net[tuple_idx++/8][feature] += delta;
		}
	}
	
	virtual void open_episode(const std::string& flag = "") {
		stats.clear();
	}

	virtual void close_episode(const std::string& flag = "") {
		int sz = stats.size();
		float delta = -1.0f * alpha * board_value(stats[sz-1].next);
		update_net(stats[sz-1].next, delta);
		for(int i=sz-1;i>=1;i--){
			float delta = alpha * (stats[i].reward + board_value(stats[i].next) - board_value(stats[i-1].next));
			update_net(stats[i-1].next, delta);
		}
	}


	virtual action take_action(const board& before) {
		float max_value = std::numeric_limits<float>::min();
		float max_reward = -1;
		int best_action = -1;

		for(auto i: {3, 2, 1, 0}){
			board after(before);
			board::reward reward = after.slide(i);
			float total = reward + board_value(after);
			if(reward != -1 and (total > max_value or max_value == std::numeric_limits<float>::min())){
				max_reward = reward;
				max_value = total;
				best_action = i;
			}
		}

		if(best_action != -1){
			board after(before);
			after.slide(best_action);
			stats.emplace_back(before, after, max_reward);
			return action::slide(best_action);
		} else {
			return action();
		}
	}


	virtual ~iso_weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	std::vector<state> stats;
	float alpha;
};

/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ?: bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};


class vanilla_greedy_slider : public random_agent {
public:
	vanilla_greedy_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		board::reward max_reward = -1;
		int max_op;
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if(reward >= max_reward){
				max_reward = reward;
				max_op = op;
			}
		}
		if(max_reward == -1){
			return action();
		} else {
			return action::slide(max_op);
		}
	}

private:
	std::array<int, 4> opcode;
};


class forbid_greedy_slider : public random_agent {
public:
	forbid_greedy_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 3 }), alter_opcode({ 2 }) {}

	virtual action take_action(const board& before) {
		board::reward max_reward = -1;
		int max_op;
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if(reward >= max_reward){
				max_reward = reward;
				max_op = op;
			}
		}
		if(max_reward == -1){
			max_reward = -1;
			for (int op : alter_opcode) {
				board::reward reward = board(before).slide(op);
				if(reward >= max_reward){
					max_reward = reward;
					max_op = op;
				}
			}
			if(max_reward != -1){
				return action::slide(max_op);
			}
			return action();
		} else {
			return action::slide(max_op);
		}
	}

private:
	std::array<int, 3> opcode;
	std::array<int, 1> alter_opcode;
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class greedy_slider : public random_agent {
public:
	greedy_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 1, 2 }), alter_opcode({3, 0}) {}

	virtual action take_action(const board& before) {
		board::reward max_reward = -1;
		int max_op;
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if(reward >= max_reward){
				max_reward = reward;
				max_op = op;
			}
		}
		if(max_reward == -1){
			max_reward = -1;
			for (int op : alter_opcode) {
				board::reward reward = board(before).slide(op);
				if(reward >= max_reward){
					max_reward = reward;
					max_op = op;
				}
			}
			if(max_reward != -1){
				return action::slide(max_op);
			}
			return action();
		} else {
			return action::slide(max_op);
		}
	}

private:
	std::array<int, 2> opcode;
	std::array<int, 2> alter_opcode;
};
