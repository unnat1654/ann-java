package ArtificialNNModel;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class BiMap<K,V>{
    Map<K,V> keyToValue;
    Map<V,K> valueToKey;

    BiMap(){
        keyToValue = new LinkedHashMap<>();
        valueToKey = new HashMap<>();
    }

    void put(K key, V value){
        if (keyToValue.containsKey(key)) {
            V oldValue = keyToValue.get(key);
            valueToKey.remove(oldValue);
        }
        if (valueToKey.containsKey(value)) {
            K oldKey = valueToKey.get(value);
            keyToValue.remove(oldKey);
        }
        keyToValue.put(key, value);
        valueToKey.put(value, key);
    }
    
    V getValue(K key) {
        return keyToValue.get(key);
    }

    K getKey(V value) {
        return valueToKey.get(value);
    }

    boolean containsKey(K key) {
        return keyToValue.containsKey(key);
    }

    int size() {
        return keyToValue.size();
    }
    
}
