#pragma once
#include <typeinfo>

namespace DeepCL
{
	namespace BackendSystem
	{
		//Base Tuple class. No constructor is sepcified, thereby the base constructor is called. It gets called when the variadic template is empty
		template <class... Ts> struct Tuple {};

		//Tuple specialization for when the tuple contains at least one element. Derives from base class with one less variadic template elements
		template<class T, class... Ts>
		struct Tuple<T, Ts...> : Tuple<Ts...>
		{
			//Calls base constructor recursivly and initalizes value. Each recursive base class constructor call is performed with one element sliced off the variadic template.
			Tuple(T t, Ts... ts) : Tuple<Ts...>(ts...), value(t) {}
			
			//Constructor when value is not initalized to any specific value
			Tuple() : Tuple<Ts...>(), value(){}

			//Stores one value of the tuple in the member value
			T value;
		};

		//Class used to retriev the type of a member in the tuple. The size_t parameter specified the index from which the type is queried
		template<size_t, class> struct ElemHolder;

		//Case when the size_t index is zero. stored the current sliced of template type in type
		template <class T, class... Ts>
		struct ElemHolder<0, Tuple<T, Ts...>>
		{
			typedef T type;
		};

		//sets the type of this derived class to be the same as the base class type recursively. The index k is reduced by one each recursive call and the first variadic template argument is sliced off.
		template <size_t k, class T, class... Ts>
		struct ElemHolder<k, Tuple<T, Ts...>>
		{
			typedef typename ElemHolder<k - 1, Tuple<Ts...>>::type type;
		};

		//The function get returns the value of the, by k specified tuple element. 
		//The ElemHolder struct is used to query the type of specific tuple element. The specified is used to set the return type of the function.
		//Base case when k equals zero
		template<size_t k, class... Ts>
		typename std::enable_if<k == 0, typename ElemHolder<0, Tuple<Ts...>>::type& >::type
			inline	get(Tuple<Ts...>& t)
		{
			return t.value;
		}

		//Case when k is not zero. It slices off an template argument each time and calls get with a by one reduced index
		template<size_t k, class T, class... Ts>
		typename std::enable_if<k != 0, typename ElemHolder<k, Tuple<T, Ts...>>::type& >::type
			inline	get(Tuple<T, Ts...>& t)
		{
			Tuple<Ts...>& base = t;
			return get<k - 1>(base);
		}
	}
}